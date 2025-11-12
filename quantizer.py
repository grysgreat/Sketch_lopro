import torch
import torch.nn as nn
import argparse
import os
import json
import numpy as np
import gc
from r1_sketch import *
from draw_fig import *

qtype = torch.float16
def find_params_group(x, bit = 4, group_size = 128, weight=False):
    dev = x.device

    maxq = torch.tensor(2 ** bit - 1)
    maxq = maxq.to(dev)

    shape = x.shape

    # use per channel
    x = x.flatten(1)

    assert x.size(0) % 128 == 0
    # 展开张量并求每128个元素的最小值
    # print(f"{x.shape}, {group_size}")
    xmin = x.unfold(1, group_size, group_size).min(dim=-1).values
    xmax = x.unfold(1, group_size, group_size).max(dim=-1).values

    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)

    # shape = [-1] + [1] * (len(shape) - 1)
    # self.scale = self.scale.reshape(shape)
    # self.zero = self.zero.reshape(shape)
    return  scale, zero


def get_scale_fp4(self, input, bits, mantissa_bit, bias):
    
    M = mantissa_bit
    E = bits - 1 - M
    bias = bias.float()
    maxval = (2 - 2 ** (-M)) * 2 ** (
            2**E - 1 - bias
        )
    minval = -maxval

    input = torch.min(torch.max(input, minval), maxval)

    input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)), 1.0)

    return input, 2.0 ** (input_log_scales - M - bias)

def quantize_lora_group(x, scale, zero, bit = 4,group=128, L = 0, R = 0, rank=0):

    dev = x.device
    maxq = torch.tensor(2 ** bit - 1)
    maxq = maxq.to(dev)

    shape = x.shape
    x_unfolded = x.unfold(1, group, group)
    q = torch.round(x_unfolded / scale.unsqueeze(-1) + zero.unsqueeze(-1))
    q = torch.clamp(q, 0, maxq)
    #q_folded = q.reshape(m, n)
    # 应用反量化公式
    x_dequantized = (q - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    
    # 将反量化后的张量重新折叠回原始形状
    x_dequantized_folded = x_dequantized.reshape(shape)
    if rank == 0:
        return x_dequantized_folded

    return x_dequantized_folded + torch.matmul(L.T,R)


def quant_sketch_save_full_process_res(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    qscale, qzero = find_params_group(W, bit = bit , group_size = groupsize,weight=True) 
    Reduce = quantize_lora_group(
        W, qscale, qzero, group=groupsize
    )
    res = W - Reduce
    feat_scale = feat_scale.cuda()
    res_scale_T = torch.diag(feat_scale) @ res.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(res_scale_T, bit, ratio = ratio, fix_rank = fix_rank)


    if srank != 0:
        lora_res = torch.matmul(r1_L.T,r1_R)
        lora_res = torch.diag(feat_scale.float()).inverse().half() @ lora_res
        Reduce = Reduce + lora_res.T
    else:
        Reduce = Reduce
    
    return Reduce,srank


def quant_sketch_save_full_process_pre(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)

    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        Reduce = 0
        
    if srank != 0:
        res = W - lora_W.T
    else:
        res = W

    # qscale, qzero = find_params_group(res, bit = bit , group_size = groupsize,weight=True) 
    # Reduce = quantize_lora_group(
    #     res, qscale, qzero, group=groupsize
    # )
    Reduce = pseudo_quantize_tensor(res, bit=bit,q_group_size=groupsize)
    if srank != 0:
        Reduce = Reduce + lora_W.T
    else:
        Reduce = Reduce
    
    return Reduce,srank



def quant_sketch_save_full_process_double(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)



    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
        res = W - lora_W.T
    else:
        res = W

    qscale, qzero = find_params_group(res, bit = bit , group_size = groupsize,weight=True) 
    Reduce = quantize_lora_group(
        res, qscale, qzero, group=groupsize
    )

    res2 = res - Reduce
    res_scale_T = torch.diag(feat_scale) @ res2.T

    W2,r1_L,r1_R,max_0,max_now,srank2 = get_best_sketch(res_scale_T, bit, ratio = ratio, fix_rank = fix_rank)

    if srank != 0:
        Reduce = Reduce + lora_W.T
    else:
        Reduce = Reduce

    if srank2 != 0:
        lora_res = torch.matmul(r1_L.T,r1_R)
        lora_res = torch.diag(feat_scale.float()).inverse().half() @ lora_res
        Reduce = Reduce + lora_res.T
    else:
        Reduce = Reduce

    return Reduce,srank



def quant_sketch_save_full_process_iter(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    qscale, qzero = find_params_group(W, bit = bit , group_size = groupsize,weight=True) 
    Reduce = quantize_lora_group(
        W, qscale, qzero, group=groupsize
    )
    res = W - Reduce
    feat_scale = feat_scale.cuda()
    res_scale_T = torch.diag(feat_scale) @ res.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(res_scale_T, bit, ratio = ratio, fix_rank = fix_rank)
    
    iteration = 32
    if srank != 0:
        lora_res = torch.matmul(r1_L.T,r1_R)
        lora_res = torch.diag(feat_scale.float()).inverse().half() @ lora_res

        qscale, qzero = find_params_group(W-lora_res.T, bit = bit , group_size = groupsize,weight=True) 
        Reduce = quantize_lora_group(
            W-lora_res.T, qscale, qzero, group=groupsize
        )    
    
        
        Reduce = Reduce + lora_res.T
    else:
        Reduce = Reduce
    
    return Reduce,srank



def sketch_pre(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128, max_sketch_iter = 4):

    dtype = W.dtype
    feat_scale = feat_scale.cuda()#.double()
    #W = W.double()


    W_scale_T = torch.diag(feat_scale) @ W.T



    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank, max_sketch_iter = max_sketch_iter)
    # W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_svd(W_scale_T, bit, ratio = ratio, fix_rank = 32)
    
    # print(W_scale_T.max())
    # srank=32
    # W_scale_T = W_scale_T.to(torch.float64)
    # U, S, Vh = torch.linalg.svd(W_scale_T, full_matrices=False)
    # # 截取前 rank 个奇异值和对应的向量
    # U_trunc = U[:, :srank]
    # S_trunc = S[:srank]
    # Vh_trunc = Vh[:srank, :]
    # lora_W = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
    
    if srank == 0:
        W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = 16, max_sketch_iter = max_sketch_iter)

    if srank != 0:
        r1_L = r1_L.float()
        r1_R = r1_R.float()
        lora_W = torch.matmul(r1_L.T,r1_R).double()
        lora_W = torch.diag(feat_scale.float()).inverse().double() @ lora_W
        # lora_W = torch.diag(feat_scale.to(torch.float64)).inverse() @ lora_W
        # lora_W = lora_W.to(torch.bfloat16)
    else:
        lora_W = 0
        
    
    return lora_W.to(dtype),srank



def sketch_pre_Test(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128, max_sketch_iter = 4):

    dtype = W.dtype
    feat_scale = feat_scale.cuda()#.double()


    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_svd(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)

    
    if srank != 0:
        r1_L = r1_L.float()
        r1_R = r1_R.float()
        lora_W = torch.matmul(r1_L.T,r1_R).double()
        lora_W = torch.diag(feat_scale.float()).inverse().double() @ lora_W

        L2 = torch.diag(feat_scale.float()).inverse().double() @ r1_L.T.double() 
        print(L2.T)
        print(r1_R)
        print(lora_W)
        print(W-lora_W.T)
    else:
        lora_W = 0
        
    
    return lora_W.to(dtype),W-lora_W.T, L2, r1_R


def sketch_pre_svd2(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128, max_sketch_iter = 4):

    dtype = W.dtype
    feat_scale = feat_scale.cuda()#.double()
    #W = W.double()


    W_scale_T = torch.diag(feat_scale) @ W.T
    srank=fix_rank
    W_scale_T = W_scale_T.to(torch.float64)
    U, S, Vh = torch.linalg.svd(W_scale_T, full_matrices=False)
    # 截取前 rank 个奇异值和对应的向量
    U_trunc = U[:, :srank]#.half()
    S_trunc = S[:srank]
    Vh_trunc = Vh[:srank, :]#.half()
    # U_parts = []
    # S_parts = []
    # Vh_parts = []

    # rank_block = 4
    # for i in range(int(srank/rank_block)):
    #     U, S, Vh = torch.linalg.svd(W_scale_T, full_matrices=False)
    #     # 截取前 rank 个奇异值和对应的向量
    #     U_trunc = U[:, :rank_block].half()
    #     S_trunc = S[:rank_block]
    #     Vh_trunc = Vh[:rank_block, :].half()        
        
    #     lora_new = U_trunc.double() @ torch.diag(S_trunc).double() @ Vh_trunc.double() 
    #     W_scale_T = W_scale_T - lora_new

    #     U_parts.append(U_trunc)
    #     S_parts.append(S_trunc)
    #     Vh_parts.append(Vh_trunc)

    # # 3. 拼接成完整矩阵
    # U_full = torch.cat(U_parts, dim=1)    # (m, total_rank)
    # S_full = torch.cat(S_parts, dim=0)    # (total_rank,)
    # Vh_full = torch.cat(Vh_parts, dim=0)  # (total_rank, n)


    lora_W = U_trunc.double() @ torch.diag(S_trunc).double() @ Vh_trunc.double() 
    lora_W = torch.diag(feat_scale.float()).inverse().double() @ lora_W

    return lora_W.to(dtype),srank


def sketch_pre_svd_split(W, feat_scale, bit=4, fix_rank=0, ratio=0.1, groupsize=128, max_sketch_iter=4, lora_bit = 16):
    dtype = W.dtype
    device = W.device  # 保存原始设备

    # 1. feat_scale 移到 GPU
    feat_scale = feat_scale.to(device)
    
    # 2. 构造 W_scale_T 并转为 float64（SVD 精度要求）
    W_scale_T = torch.diag(feat_scale) @ W.T
    W_scale_T = W_scale_T.to(torch.float64)

    # 3. 执行 SVD
    U, S, Vh = torch.linalg.svd(W_scale_T, full_matrices=False)

    srank = fix_rank

    # 4. 截取前 srank 个分量，并立即移到 CPU
    U_trunc = U[:, :srank].detach().cpu()
    S_trunc = S[:srank].detach().cpu()
    Vh_trunc = Vh[:srank, :].detach().cpu()

    # 5. feat_scale 倒数也移到 CPU
    Sa = (torch.tensor(1.0, dtype=torch.float32)/feat_scale.float())

    # 6. 构造返回结构
    if lora_bit == 8:
        lora_struct = {
            "U": U_trunc.cuda().to(torch.float8_e4m3fn),
            "V": Vh_trunc.cuda().to(torch.float8_e4m3fn),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }
    else:
        lora_struct = {
            "U": U_trunc.cuda().to(dtype),
            "V": Vh_trunc.cuda().to(dtype),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }        

    # 7. 删除所有 GPU 中间变量
    del W_scale_T, U, S, Vh, feat_scale
    del U_trunc, S_trunc, Vh_trunc, Sa  # 可选：这些已经是 CPU，但引用可删

    # 8. 手动清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保所有操作完成
        torch.cuda.empty_cache()  # 释放缓存显存

    return lora_struct, srank



def sketch_pre_split(W, feat_scale, bit=4, fix_rank=0, ratio=0.1, groupsize=128, max_sketch_iter=4, lora_bit = 16, lora_iter = 8):
    dtype = W.dtype
    device = W.device  # 保存原始设备

    # 1. feat_scale 移到 GPU
    feat_scale = feat_scale.to(device)
    
    # 2. 构造 W_scale_T 并转为 float64（SVD 精度要求）
    W_scale_T = torch.diag(feat_scale) @ W.T
    W_scale_T = W_scale_T.to(torch.float64)

    # # 3. 执行 SVD
    # U, S, Vh = torch.linalg.svd(W_scale_T, full_matrices=False)

    # srank = fix_rank

    # # 4. 截取前 srank 个分量，并立即移到 CPU
    # U_trunc = U[:, :srank].detach().cpu()
    # S_trunc = S[:srank].detach().cpu()
    # Vh_trunc = Vh[:srank, :].detach().cpu()



    W2,U_trunc,Vh_trunc,S_trunc,max_0,max_now,srank = get_best_sketch_fp8_ret(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank, max_sketch_iter = lora_iter)
    

    U_trunc = [tensor.to(torch.float16) for tensor in U_trunc]
    Vh_trunc = [tensor.to(torch.float16) for tensor in Vh_trunc]

    U_trunc = torch.vstack(U_trunc[:srank])
    Vh_trunc = torch.vstack(Vh_trunc[:srank])
    S_trunc = torch.tensor(S_trunc).cuda()
    # 5. feat_scale 倒数也移到 CPU
    Sa = (torch.tensor(1.0, dtype=torch.float32)/feat_scale.float())



    # 6. 构造返回结构
    if lora_bit == 8:
        lora_struct = {
            "U": U_trunc.cuda().T,
            "V": Vh_trunc.cuda(),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }
    else:
        lora_struct = {
            "U": U_trunc.cuda().to(dtype).T,
            "V": Vh_trunc.cuda().to(dtype),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }        

    # 7. 删除所有 GPU 中间变量
    del W_scale_T,  feat_scale
    del U_trunc, S_trunc, Vh_trunc, Sa  # 可选：这些已经是 CPU，但引用可删

    # 8. 手动清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保所有操作完成
        torch.cuda.empty_cache()  # 释放缓存显存

    return lora_struct, srank




def sketch_pre_svd(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128, max_sketch_iter = 4):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    #W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank, max_sketch_iter = max_sketch_iter)
    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_svd(W_scale_T, bit, ratio = ratio, fix_rank = 32)
    
    

    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().to(qtype) @ lora_W
    else:
        lora_W = 0
        
    
    return lora_W,srank

def sketch_pre_fp8(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_fp8(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)
    #W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_svd(W_scale_T, bit, ratio = ratio, fix_rank = 32)
    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        lora_W = 0
        
    
    return lora_W,srank

def sketch_pre_fp8_ret(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,S_arr,max_0,max_now,srank = get_best_sketch_fp8_ret(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)
    #W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_svd(W_scale_T, bit, ratio = ratio, fix_rank = 32)
    if srank != 0:
        r1_L_16 = [tensor.to(torch.float16) for tensor in r1_L]
        r1_R_16 = [tensor.to(torch.float16) for tensor in r1_R]

        r1_L_16 = torch.vstack(r1_L_16[:srank])
        r1_R_16 = torch.vstack(r1_R_16[:srank])

        r1_L_16 = r1_L_16.T @ torch.diag(torch.tensor(S_arr).cuda().half())
        r1_L_16 = r1_L_16.T
        lora_W = torch.matmul(r1_L_16.T,r1_R_16)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        lora_W = 0
        
    
    return lora_W, srank,r1_L,r1_R,S_arr


def sketch_pre_full_ret(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)
    #W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_svd(W_scale_T, bit, ratio = ratio, fix_rank = 32)
    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        lora_W = 0
        
    
    return lora_W,srank,r1_L,r1_R



def sketch_pre_diff(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_group_diff_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)

    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        lora_W = 0
        
    
    return lora_W,srank

def sketch_pre_mse(W, w_q ,w_org,  feat_scale, input_feat, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_mse_best_sketch(W_scale_T, w_q ,w_org,  feat_scale, input_feat, bit, ratio = ratio, fix_rank = fix_rank)

    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        lora_W = 0
    return lora_W,srank

def pseudo_quantize_tensor0(w,bit =4,group_size=0):
    qscale, qzero = find_params_group(w, bit = bit , group_size = group_size,weight=True) 
    qw = quantize_lora_group(
        w, qscale, qzero, group=group_size
    )
    return qw


def sketch_pre_print_mse(W, feat_scale, input_feat, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128, name = "default"):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T
    

    W2,r1_L,r1_R,max_0,max_now,srank0 = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = 0)

    W2,r1_L,r1_R,max_0,max_now,srank,sVals = get_best_sketch_retS(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)


    max_now = find_max_abs_value(W)
    group_max_now = find_group_max_diff_value(W,0)
    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        Reduce = 0
        
    input_feat = input_feat.cuda()
    scale_org = input_feat @ W.T


    Reduce = pseudo_quantize_tensor(W, bit=bit,q_group_size=groupsize)
    scale_out = (input_feat @ (Reduce.T))
    err = (scale_out - scale_org).float().pow(2).sum().item()
    rela_err = err / (scale_org.float().pow(2).sum().item())

    
    rela_array = np.zeros(srank+1)
    max_now_array = np.zeros(srank+1)
    #print(rela_err,max_now.cpu().item())
    rela_array[0] = rela_err
    max_now_array[0] = max_now.cpu().item()

    res = W
    lora_now = torch.zeros_like(res)

    for i in range(srank):
        lora_1 = torch.ger(r1_L[i],r1_R[i])
        lora_1 = torch.diag(feat_scale.float()).inverse().half() @ lora_1
        lora_now = lora_now + lora_1.T
        res = res - lora_1.T
        max_now = find_max_abs_value(res)
        group_max_now = find_group_max_diff_value(res,groupsize)
        Reduce = pseudo_quantize_tensor(res, bit=bit,q_group_size=groupsize)
        scale_out = (input_feat @ (Reduce.T + lora_now.T))
        err = (scale_out - scale_org).float().pow(2).sum().item()
        rela_err = err / (scale_org.float().pow(2).sum().item())
        exb = 16*i*(W.size(0)+W.size(1)+0.01)/(W.size(0)*W.size(1))
        #print(rela_err,max_now.cpu().item(),group_max_now,exb)
        rela_array[i+1] = rela_err
        max_now_array[i+1] = max_now.cpu().item()
    x = list(range(0, srank))
    draw_mse(x,sVals,rela_array[0:srank],srank0,name)
    return lora_W,srank

def sketch_pre_mse_ret(W, feat_scale, input_feat, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128, name = "default"):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T
    

    W2,r1_L,r1_R,max_0,max_now,srank0 = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = 0)

    W2,r1_L,r1_R,max_0,max_now,srank,sVals = get_best_sketch_retS(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)


    max_now = find_max_abs_value(W)
    group_max_now = find_group_max_diff_value(W,0)
    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().half() @ lora_W
    else:
        Reduce = 0
        
    input_feat = input_feat.cuda()
    scale_org = input_feat @ W.T


    Reduce = pseudo_quantize_tensor(W, bit=bit,q_group_size=groupsize)
    scale_out = (input_feat @ (Reduce.T))
    err = (scale_out - scale_org).float().pow(2).sum().item()
    rela_err = err / (scale_org.float().pow(2).sum().item())

    
    rela_array = np.zeros(srank+1)
    max_now_array = np.zeros(srank+1)
    #print(rela_err,max_now.cpu().item())
    rela_array[0] = rela_err
    max_now_array[0] = max_now.cpu().item()

    res = W
    lora_now = torch.zeros_like(res)

    for i in range(srank):
        lora_1 = torch.ger(r1_L[i],r1_R[i])
        lora_1 = torch.diag(feat_scale.float()).inverse().half() @ lora_1
        lora_now = lora_now + lora_1.T
        res = res - lora_1.T
        max_now = find_max_abs_value(res)
        group_max_now = find_group_max_diff_value(res,groupsize)
        Reduce = pseudo_quantize_tensor(res, bit=bit,q_group_size=groupsize)
        scale_out = (input_feat @ (Reduce.T + lora_now.T))
        err = (scale_out - scale_org).float().pow(2).sum().item()
        rela_err = err / (scale_org.float().pow(2).sum().item())
        exb = 16*i*(W.size(0)+W.size(1)+0.01)/(W.size(0)*W.size(1))
        #print(rela_err,max_now.cpu().item(),group_max_now,exb)
        rela_array[i+1] = rela_err
        max_now_array[i+1] = max_now.cpu().item()
    x = list(range(0, srank))
    #draw_mse(x,sVals,rela_array[0:srank],srank0,name)
    return lora_W,srank, sVals, rela_array[0:srank]



# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, bit=8, q_group_size=-1
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)


    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0


    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w


def pseudo_quantize_tensor_do(
    w, bit=8, q_group_size=-1
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    rezeros = zeros * scales

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0


    w = (
        torch.clamp(torch.round((w + rezeros) / scales), min_int, max_int) - zeros
    ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w


# core quantization method (simulated quantization)
def pseudo_quantize_tensor_2bit(
    w, bit=8, q_group_size=-1
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2


    w = process_tensor(w)
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w

def process_tensor(tensor):
   # 1. 对每个行进行排序
    sorted_tensor, _ = tensor.sort(dim=1)
    
    # 2. 提取四个分组的中间元素作为中点
    points_indices = [15, 47, 79, 111]  # 每个分组的中间索引（0-based）
    points = sorted_tensor[:, points_indices]  # 形状 (N, 4)
    
    # 3. 扩展points的维度以便与原始张量广播
    points_expanded = points.unsqueeze(1).expand(-1, tensor.size(1), -1)
    # 扩展后形状为 (N, 128, 4)
    
    # 4. 扩展原始张量的维度
    tensor_expanded = tensor.unsqueeze(2)  # 形状 (N, 128, 1)
    
    # 5. 计算每个元素到四个中点的距离
    differences = torch.abs(tensor_expanded - points_expanded)
    # 形状 (N, 128, 4)
    
    # 6. 找到最近中点的索引（0到3）
    min_indices = torch.argmin(differences, dim=2)  # 形状 (N, 128)
    
    # 7. 根据索引选择对应的中点值
    # 将索引转换为unsqueeze后的形状 (N, 128, 1)
    selected_points = torch.gather(
        points_expanded,  # 形状 (N, 128, 4)
        dim=2,           # 沿着第三个维度（中点索引）
        index=min_indices.unsqueeze(2)  # 形状 (N, 128, 1)
    ).squeeze(2)         # 最终形状 (N, 128)
    
    return selected_points