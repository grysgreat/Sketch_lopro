import numpy as np
import matplotlib.pyplot as plt
import platform
from numpy import random
import time
from contextlib import contextmanager
import torch
import math
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import re
import json
from transformers import AutoTokenizer, TextGenerationPipeline
import logging
import shutil
# model_file_path =  '/datas/opt-125m'
# output_file_path =  '/datas/opt-125m-skqlora'
# model_name = "opt-125m"
# group = '128'


# qbit = '4'
# fix_rank = 0
# ratio = 0.2
@contextmanager
def timing_context(name):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{name} executed in {execution_time:.4f} seconds")


def plot_weight_histogram(weights_np, name,bin_width=0.005):    
    # 计算直方图的区间

    weights_np = weights_np.to(torch.float16).cpu().numpy()
    min_val = np.min(weights_np)
    max_val = np.max(weights_np)
    bins = np.arange(min_val, max_val + bin_width, bin_width)


    max_abs_value = np.max(np.abs(weights_np))
    print(f"min '{min_val}' max {max_val}  max2 {max_abs_value} shape {weights_np.shape}")

    # 绘制直方图
    plt.hist(weights_np.flatten(), bins=bins, edgecolor='black')
    
    # 设置坐标轴标签
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Frequency')
    
    # 设置标题
    plt.title('Histogram of Weight Magnitudes in the First Layer')
    plt.savefig("/home/ghyx/tmp/fig/"+name+'.png', dpi=300, bbox_inches='tight')
    # 显示图形
    plt.close()


def find_max_abs_value(A):
    # 计算矩阵 A 的绝对值
    abs_A = torch.abs(A)
    # 找到绝对值最大的数
    max_abs_value = torch.max(abs_A)
    return max_abs_value

def compute_r1sketch(A,iter = 1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = A.shape

    x_numpy = random.normal(loc=0, scale=1, size=(n))
    # 生成一个长度为 n 的随机正态分布向量 x
    
    x = torch.from_numpy(x_numpy)
    x = x.to(torch.float32)
    x = x.to(device)
    y = torch.matmul(A, x)

    for i in range(iter):
        tmp = torch.matmul(A.T, y)
        y = torch.matmul(A, tmp)


    A_L = y
    A_R = torch.matmul(A.T, A_L)


    normP = torch.norm(A_L, p=2)
    normQ = torch.norm(A_R, p=2)

    Var_AL = normQ/(normP*normP)
    Var_AR = 1.0/normQ


    A_R = A_R*Var_AR
    A_L = A_L*Var_AL
    # SK = np.outer(A_L, A_R)
    return A_L, A_R



def get_best_sketch(weights, bits, ratio=0.01, max_sketch_iter = 3, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float32)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)

    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = max(max_ptr//4,4)
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L,r1_R = compute_r1sketch(weights,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
        max_now = max_iter[work_rank-1]
    else:
        for i in range(0,min_rank):
            r1_L,r1_R = compute_r1sketch(weights,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            P = max_absW_0/max_absW
            K = 1.0+(16.0*(i+1)*(row+col)/(1.0*bits*row*col))
            Q = (bits + math.log(P,2) )/(1.0*bits)
            #print(f"max_absW: {max_absW}, rank: {i+1},K: {K}, Q:{Q}")
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            if(i>=max_ptr):
                if(((max_iter[i-max_ptr]-max_iter[i])/max_iter[i-max_ptr])<0.02):
                    work_rank = i-max_ptr//2
                    break

            if(K>(1.0+ratio)):
                work_rank = i
                break

            if (K >= Q and i > max_sketch_iter+min_ptr):
                work_rank = i - 1
                break
            else:
                work_rank = i

        
        if(work_rank == 0):
            work_rank = 1
        if (max_absW_0 - max_iter[work_rank-1])/max_absW_0 < 0.1:
            work_rank = min_ptr
            max_now = max_absW_0

        if(work_rank>=1):
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
            max_now = max_iter[work_rank-1]

        if(work_rank!=0 and max_absW_0<=max_iter[work_rank-1]):
            work_rank = 0
            VS_L.zero_()
            VS_R.zero_()
            max_now = max_absW_0

    #plot_weight_histogram(weights, "W2")
    if work_rank!=0:
        VS_L_16 = VS_L.to(torch.float16)
        VS_R_16 = VS_R.to(torch.float16)
        weight_cp = weight_cp - torch.matmul(VS_L_16.T,VS_R_16)
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank
    #return weight_cp,VS_L,VS_R,max_absW_0,max_iter[work_rank-1]


def get_best_sketch_svd(weights, bits, ratio=0.01, max_sketch_iter = 4, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float32)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)
    U, s, Vt = torch.linalg.svd(weights, full_matrices=True)
    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L = U[:, i].reshape(-1)* s[i]
            r1_R = Vt[i, :].reshape(-1)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
        max_now = max_iter[work_rank-1]


    #plot_weight_histogram(weights, "W2")
    if work_rank!=0:
        VS_L_16 = VS_L.to(torch.float16)
        VS_R_16 = VS_R.to(torch.float16)
        weight_cp = weight_cp - torch.matmul(VS_L_16.T,VS_R_16)
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank
    #return weight_cp,VS_L,VS_R,max_absW_0,max_iter[work_rank-1]




def load_model(model_path):
    # 判断文件类型
    if model_path.endswith('.pth'):
        # 加载 .pth 文件
        print("Loading .pth model")
        model = torch.load(model_path)
    elif model_path.endswith('.safetensors'):
        # 加载 .safetensor 文件
        print("Loading .safetensor model")
        model = load_file(model_path)  # 使用 safetensors 的 load_file
    else:
        raise ValueError("Unsupported model file format")
    
    return model

def quant_sketch(file_path, output_file_path, bit = 4):
    # 加载模型权重
    model_weights = load_model(file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sketch_data = {}
    # 打开输出文件
    for layer_name, weights in model_weights.items():

        if weights.dim() > 1:
            # 如果权重是bf16格式，转换为float32以进行计算
            if weights.dtype == torch.bfloat16:
                weights = weights.to(torch.float32)
            if weights.dtype == torch.float16:
                weights = weights.to(torch.float32)
            
            weights = weights.to(device)
            W = weights
            #plot_weight_histogram(W, "W")
            #print(f"layer:{layer_name}")
            W2,r1_L,r1_R,max_0,max_now = get_best_sketch(W, bit, 0.05)

            #plot_weight_histogram(W2, "W2")
            sketch_L = layer_name + "_L"
            sketch_R = layer_name + "_R"               
            sketch_data[sketch_L] = r1_L
            sketch_data[sketch_R] = r1_R
            print(f"{layer_name}, rank: {np.size(r1_L,0)} ,{W.shape}, m0:{max_0}, mnow:{max_now}")

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


def find_params_mini(self, x):
    dev = x.device

    maxq = torch.tensor(2 ** bit - 1)
    maxq = maxq.to(dev)

    shape = x.shape

    x = x.flatten(1)

    tmp = torch.zeros(x.shape[0], device=dev)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1


    scale = (xmax - xmin) / maxq

    zero = torch.round(-xmin / scale)

    shape = [-1] + [1] * (len(shape) - 1)
    scale =  scale.reshape(shape)
    zero =  zero.reshape(shape)
    return scale, zero

def quantize_group(x, scale, zero, bit = 4, group=128):

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

    return x_dequantized_folded


def quantize_lora_group(x, scale, zero,L,R,rank,bit = 4,group=128):

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


def quant_sketch_save_full_process(model_weights, output_file_path, bit = 4, metadata = None, fix_rank = 0, ratio = 0.1,groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight' and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)
                W = weights.to(device)

                W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W, bit, ratio = ratio, fix_rank = fix_rank)

                # W2_,r1_L_,r1_R_,max_0_,max_now_,srank_ = get_best_sketch(W, bit, ratio = ratio, fix_rank = fix_rank)


                # if max_now_ < max_now:
                #     W2 = W2_
                #     max_now = max_now_
                #     r1_L = r1_L_
                #     r1_R = r1_R_
                #     srank = srank_
                #plot_weight_histogram(W2, "W2")

                if srank != 0:
                    Reduce = W - torch.matmul(r1_L.T,r1_R)
                else:
                    Reduce = W

                if(groupsize==0):
                    new_group = Reduce.size(1)
                else:
                    new_group = groupsize
                scale,zero = find_params_group(Reduce, bit = bit , group_size = new_group,weight=True) 
                Reduce = quantize_lora_group(
                    Reduce, scale, zero,  r1_L, r1_R, srank, group=new_group
                )
                reduce_fp16_data[layer_name] = Reduce.to(torch.device("cpu"))#weights.to(torch.float16)

                org_size = org_size+ weights.size(0)* weights.size(1)*16
                sketch_size = sketch_size+(weights.size(0)+weights.size(1))*srank*16
                quant_size = quant_size+weights.size(0)* weights.size(1)*bit + ((16*weights.size(0)* weights.size(1))/new_group)


                total_rank += srank
                layer_cnt = layer_cnt + 1

                if info:
                    print(f"{layer_name}, rank: {srank} ,{W.shape}, m0:{max_0}, mnow:{max_now}")

            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights

    q_bit = 16.0*(float(quant_size)/float(org_size))
    loraq_bit = 16.0*(float(quant_size+sketch_size)/float(org_size))
    print(f"total rank = {total_rank}, sketch_num = {layer_cnt}, avg_rank = {(total_rank+0.001)/(layer_cnt+0.001)}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)


def quant_sketch_save_full_process_mask(model_weights, output_file_path, bit = 4, metadata = None, fix_rank = 0, ratio = 0.1,groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight' and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)
                W = weights.to(device)
                tmpnum=0
                W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W, bit, ratio = ratio, fix_rank = fix_rank)

                # W2_,r1_L_,r1_R_,max_0_,max_now_,srank_ = get_best_sketch(W, bit, ratio = ratio, fix_rank = fix_rank)


                # if max_now_ < max_now:
                #     W2 = W2_
                #     max_now = max_now_
                #     r1_L = r1_L_
                #     r1_R = r1_R_
                #     srank = srank_
                #plot_weight_histogram(W2, "W2")

                xmin0 = W.unfold(1, groupsize, groupsize).min(dim=-1).values
                xmax0 = W.unfold(1, groupsize, groupsize).max(dim=-1).values
                xscale0 = xmax0 - xmin0
                means0 = torch.mean(xscale0)
                if srank != 0:
                    scale,zero = find_params_group(W, bit = bit , group_size = groupsize,weight=True) 
                    Reduce = quantize_lora_group(
                        W, scale, zero,  r1_L, r1_R, tmpnum, group=groupsize
                    )
                    red = torch.abs(W - Reduce)
                    error1 = abs(red.to(torch.float64).sum().item())

                    Reduce = W - torch.matmul(r1_L.T,r1_R)

                    scale,zero = find_params_group(Reduce, bit = bit , group_size = groupsize,weight=True) 
                    Reduce = quantize_lora_group(
                        Reduce, scale, zero,  r1_L, r1_R, tmpnum, group=groupsize
                    )
                    red2 = torch.abs(W - Reduce + torch.matmul(r1_L.T,r1_R))
                    error2 = abs(red2.to(torch.float64).sum().item())

                    print(f"before:{r1_L.size(0)}:  {error1}, {error2}")

                else:
                    Reduce = W



                xmin1 = Reduce.unfold(1, groupsize, groupsize).min(dim=-1).values
                xmax1 = Reduce.unfold(1, groupsize, groupsize).max(dim=-1).values
                xscale1 = xmax1 - xmin1

                
                mask = xscale1/xscale0
                

                # 将小于 0 的元素设置为 1
                mask[mask <= 1] = 0
                mask[mask > 1] = 1
                mask = torch.logical_not(mask).to(torch.int) 
                mask_exp = mask.unsqueeze(-1).expand(*mask.shape, groupsize).reshape(mask.size(0), mask.size(1) * groupsize)                
                positive_count = (mask == 0).sum().item()
                negative_count = (mask == 1 ).sum().item()

                print(f"before: {layer_name}, {srank}, {positive_count}, {negative_count}, {mask.shape}")

                if srank != 0:
                    scale,zero = find_params_group(W, bit = bit , group_size = groupsize,weight=True) 
                    Reduce = quantize_lora_group(
                        W, scale, zero,  r1_L, r1_R, tmpnum, group=groupsize
                    )
                    red = torch.abs(W - Reduce)
                    error1 = abs(red.to(torch.float64).sum().item())

                    skmatrix = torch.matmul(r1_L.T,r1_R)
                    skmask = torch.mul(skmatrix, mask_exp)
                    Reduce = W - skmask

                    scale,zero = find_params_group(Reduce, bit = bit , group_size = groupsize,weight=True) 
                    Reduce = quantize_lora_group(
                        Reduce, scale, zero,  r1_L, r1_R, tmpnum, group=groupsize
                    )
                    red2 = torch.abs(W - Reduce + skmask)
                    error2 = abs(red2.to(torch.float64).sum().item())
                    
                    print(f"after:{r1_L.size(0)}:  {error1}, {error2}")


                else:
                    Reduce = W                

                xmin2 = Reduce.unfold(1, groupsize, groupsize).min(dim=-1).values
                xmax2 = Reduce.unfold(1, groupsize, groupsize).max(dim=-1).values
                xscale2 = xmax2 - xmin2

                
                mask2 = xscale2 - xscale0
                mask2[mask2 > 0] = 1

                # 将小于 0 的元素设置为 1
                mask2[mask2 <= 0] = 0
                mask2 = torch.logical_not(mask2).to(torch.int) 
                positive_count2 = (mask2 == 0).sum().item()
                negative_count2 = (mask2 == 1 ).sum().item()

                #print(f"after: {layer_name}, {srank}, {positive_count2}, {negative_count2}, {mask.shape}")



                scale,zero = find_params_group(Reduce, bit = bit , group_size = groupsize,weight=True) 


                stmp = srank
                srank = 0
                
                #print(f"{layer_name}, {srank}, {torch.max(weights)}|{torch.max(Reduce_tensor)}, {torch.min(weights)}|{torch.min(Reduce_tensor)},")
                Reduce = quantize_lora_group(
                    Reduce, scale, zero,  r1_L, r1_R, srank, group=groupsize
                )
                if stmp != 0:
                    Reduce = Reduce + skmask
                # if srank != 0:
                #     assert r1_L.size(0) == srank
                #     Reduce_tensor = Reduce_tensor + torch.matmul(r1_L.T,r1_R)

                reduce_fp16_data[layer_name] = Reduce.to(torch.device("cpu"))#weights.to(torch.float16)

                #print(f"{layer_name}, {max_0},{torch.max(torch.abs(Reduce_tensor))}")

                org_size = org_size+ weights.size(0)* weights.size(1)*16
                sketch_size = sketch_size+(weights.size(0)+weights.size(1))*srank*16
                quant_size = quant_size+weights.size(0)* weights.size(1)*bit + ((16*weights.size(0)* weights.size(1))/groupsize)


                total_rank += srank
                layer_cnt = layer_cnt + 1

                if info:
                    print(f"{layer_name}, rank: {srank} ,{W.shape}, m0:{max_0}, mnow:{max_now}")
                # print(f"{max_now}")
                # # 保存更新后的字典回 Safe Tensor 文件

            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights

    q_bit = 16.0*(float(quant_size)/float(org_size))
    loraq_bit = 16.0*(float(quant_size+sketch_size)/float(org_size))
    print(f"total rank = {total_rank}, sketch_num = {layer_cnt}, avg_rank = {(total_rank+0.001)/(layer_cnt+0.001)}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)



def quant_sketch_save_full_process_res(model_weights, output_file_path, bit = 4, metadata = None, fix_rank = 0, ratio = 0.1,groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight' and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)
                W = weights.to(device)
                tmp0 = 0
                r1_L=0
                r1_R=0
                scale,zero = find_params_group(W, bit = bit , group_size = groupsize,weight=True) 
                Reduce = quantize_lora_group(
                    W, scale, zero,  r1_L, r1_R, tmp0, group=groupsize
                )

                res = W - Reduce

                W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(res, bit, ratio = ratio, fix_rank = fix_rank)

                if srank != 0:
                    Reduce = Reduce + torch.matmul(r1_L.T,r1_R)
                else:
                    Reduce = Reduce


                reduce_fp16_data[layer_name] = Reduce.to(torch.device("cpu"))#weights.to(torch.float16)

                org_size = org_size+ weights.size(0)* weights.size(1)*16
                sketch_size = sketch_size+(weights.size(0)+weights.size(1))*srank*16
                quant_size = quant_size+weights.size(0)* weights.size(1)*bit + ((16*weights.size(0)* weights.size(1))/groupsize)


                total_rank += srank
                layer_cnt = layer_cnt + 1

                if info:
                    print(f"{layer_name}, rank: {srank} ,{W.shape}, m0:{max_0}, mnow:{max_now}")

            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights

    q_bit = 16.0*(float(quant_size)/float(org_size))
    loraq_bit = 16.0*(float(quant_size+sketch_size)/float(org_size))
    print(f"total rank = {total_rank}, sketch_num = {layer_cnt}, avg_rank = {(total_rank+0.001)/(layer_cnt+0.001)}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)

def quant_sketch_save_full_process_double(model_weights, output_file_path, bit = 4, metadata = None, fix_rank = 0, ratio = 0.1,groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight' and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)
                W = weights.to(device)

                W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W, bit, ratio = ratio, fix_rank = fix_rank)
                if srank != 0:
                    # w_128 = W[0, :128]
                    # plot_weight_histogram(w_128, layer_name+"bef")
                    Reduce = W - torch.matmul(r1_L.T,r1_R)
                    # w_128 = Reduce[0, :128]
                    # plot_weight_histogram(w_128, layer_name+"aft")

                else:
                    Reduce = W
                


                scale,zero = find_params_group(Reduce, bit = bit , group_size = groupsize,weight=True) 
                QReduce = quantize_lora_group(
                    Reduce, scale, zero,  r1_L, r1_R, srank, group=groupsize
                )

                res = Reduce - QReduce

                W3,r1_L2,r1_R2,max_02,max_now2,srank2 = get_best_sketch(res, bit, ratio = ratio, fix_rank = fix_rank)

                if srank != 0:
                    Reduce = QReduce + torch.matmul(r1_L.T,r1_R)
                else:
                    Reduce = QReduce
                if srank2 != 0:
                    Reduce = Reduce + torch.matmul(r1_L2.T,r1_R2)
                else:
                    Reduce = Reduce

                reduce_fp16_data[layer_name] = Reduce.to(torch.device("cpu"))#weights.to(torch.float16)

                org_size = org_size+ weights.size(0)* weights.size(1)*16
                sketch_size = sketch_size+(weights.size(0)+weights.size(1))*(srank+srank2)*16
                quant_size = quant_size+weights.size(0)* weights.size(1)*bit + ((16*weights.size(0)* weights.size(1))/groupsize)


                total_rank += srank
                layer_cnt = layer_cnt + 1

                if info:
                    print(f"{layer_name}, rank: {srank} ,{W.shape}, m0:{max_0}, mnow:{max_now}")

            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights

    q_bit = 16.0*(float(quant_size)/float(org_size))
    loraq_bit = 16.0*(float(quant_size+sketch_size)/float(org_size))
    print(f"total rank = {total_rank}, sketch_num = {layer_cnt}, avg_rank = {(total_rank+0.001)/(layer_cnt+0.001)}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)


def quant_RTN_save_full_process(model_weights, output_file_path, bit = 4, metadata = None, groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight'  and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)

                weights = weights.to(device)
                if(groupsize==0):
                    new_group = weights.size(1)
                else:
                    new_group = groupsize
                #print(f"{weights.shape}, {groupsize}, {new_group}")
                scale,zero = find_params_group(weights, bit = bit , group_size = new_group,weight=True) 

                r1_L=0
                r1_R=0
                srank=0


                weights = quantize_lora_group(
                    weights, scale, zero,  r1_L, r1_R, srank, group=new_group
                )


                reduce_fp16_data[layer_name] = weights.to(torch.device("cpu"))#weights.to(torch.float16)
            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights
    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)


def quant_sketch_save_full_process_iter(model_weights, output_file_path, bit = 4, metadata = None,fix_rank = 0,ratio = 0.1, groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight'  and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)

                weights = weights.to(device)
                if(groupsize==0):
                    new_group = weights.size(1)
                else:
                    new_group = groupsize

                W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(weights, bit, ratio = ratio, fix_rank = fix_rank)
                #print(f"{weights.shape}, {groupsize}, {new_group}")
                iteration = 32
                rm = weights

                if srank!=0:
                    for i in range(iteration):
                        W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(rm, bit, ratio = ratio, fix_rank = srank)
                        Reduce = weights - torch.matmul(r1_L.T,r1_R)

                        scale,zero = find_params_group(Reduce, bit = bit , group_size = new_group,weight=True) 
                        Reduce = quantize_group(
                            Reduce, scale, zero,   group=new_group
                        )
                        rm = weights - Reduce
                    weights = Reduce + torch.matmul(r1_L.T,r1_R)
                else:
                    scale,zero = find_params_group(weights, bit = bit , group_size = new_group,weight=True) 
                    weights = quantize_group(
                        weights, scale, zero,   group=new_group
                    )                    
                if info:
                    print(f"{layer_name}, rank: {srank} ,{W.shape}, m0:{max_0}, mnow:{max_now}")
                
                reduce_fp16_data[layer_name] = weights.to(torch.device("cpu"))#weights.to(torch.float16)
            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights
    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)

def quant_sketch_only_save_full_process(model_weights, output_file_path, bit = 4, metadata = None, fix_rank = 0, ratio = 0.1,groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight' and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)
                W = weights.to(device)

                W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch_svd(W, bit, ratio = ratio, fix_rank = fix_rank)

                Reduce = torch.matmul(r1_L.T,r1_R)
                reduce_fp16_data[layer_name] = Reduce.to(torch.device("cpu"))#weights.to(torch.float16)

                org_size = org_size+ weights.size(0)* weights.size(1)*16
                sketch_size = sketch_size+(weights.size(0)+weights.size(1))*srank*16
                quant_size = quant_size+weights.size(0)* weights.size(1)*bit + ((16*weights.size(0)* weights.size(1))/groupsize)


                total_rank += srank
                layer_cnt = layer_cnt + 1

                if info:
                    print(f"{layer_name}, rank: {srank} ,{W.shape}, m0:{max_0}, mnow:{max_now}")

            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights

    q_bit = 16.0*(float(quant_size)/float(org_size))
    loraq_bit = 16.0*(float(quant_size+sketch_size)/float(org_size))
    print(f"total rank = {total_rank}, sketch_num = {layer_cnt}, avg_rank = {(total_rank+0.001)/(layer_cnt+0.001)}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)


def quant_sketch_only_quant_save_full_process(model_weights, output_file_path, bit = 4, metadata = None, fix_rank = 0, ratio = 0.1,groupsize = 128, info = False):
    # 加载模型权重
    #model_weights = torch.load(file_path, map_location='cpu')  # 将模型加载到CPU上
    sketch_data = {}
    reduce_data = {}
    reduce_fp16_data = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rank  = 0
    layer_cnt = 0

    org_size = 0
    sketch_size = 0
    quant_size = 0
    # 打开输出文件
    for layer_name, weights in model_weights.items():
        if layer_name!='lm_head.weight' and layer_name!='model.embed_tokens.weight' and layer_name!= 'model.decoder.embed_positions.weight' and layer_name != 'model.decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_tokens.weight' and layer_name!= 'decoder.embed_positions.weight' and layer_name!= 'word_embeddings.weight':
            if weights.dim() > 1:
                if weights.dtype == torch.float32:
                    weights = weights.to(torch.float16)
                W = weights.to(device)


                r1_L=0
                r1_R=0
                srank=0
                weights = weights.to(torch.float32)
                U, s, Vt = torch.linalg.svd(weights, full_matrices=False)
                scale,zero = find_params_group(U, bit = bit , group_size = groupsize,weight=True) 
                U = quantize_lora_group(
                    U, scale, zero,  r1_L, r1_R, srank, group=groupsize
                )                

                scale,zero = find_params_group(Vt, bit = bit , group_size = groupsize,weight=True) 
                Vt = quantize_lora_group(
                    Vt, scale, zero,  r1_L, r1_R, srank, group=groupsize
                )   
                Sigma = torch.diag(s)

                #print(f"{U.shape}, {Sigma.shape}, {Vt.shape}")
                Reduce = U @ Sigma @ Vt

                Reduce = Reduce.to(torch.float16)

                reduce_fp16_data[layer_name] = Reduce.to(torch.device("cpu"))#weights.to(torch.float16)
            else:
                reduce_fp16_data[layer_name] = weights
        else:
            reduce_fp16_data[layer_name] = weights


    reduce_path = output_file_path+"/model.safetensors"

    with timing_context("Save file"):
        save_file(reduce_fp16_data, reduce_path, metadata=metadata)



def merge_weight(model_weights1,model_weights2):
    # 创建一个新的字典来存储合并后的权重
    combined_weights = {}

    # 合并第一个模型的权重
    for key, value in model_weights1.items():
        combined_weights[key] = value  # 直接添加到新的字典中
    # 合并第二个模型的权重，处理键冲突
    for key, value in model_weights2.items():
        if key in combined_weights:
            print(f"warning: key '{key}' already exist!")
            # new_key = f"{key}_2"  # 如果有重复的键，给新键添加后缀
            # combined_weights[new_key] = value
        else:
            combined_weights[key] = value  # 如果没有冲突，直接添加

    return combined_weights

def copy_small_files(src_folder, dst_folder, max_size_mb=10, info = False):
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 遍历源文件夹及其子文件夹
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if file_name!= 'model.safetensors.index.json':
                src_file_path = os.path.join(root, file_name)
                # 获取文件大小，单位为字节
                file_size = os.path.getsize(src_file_path)
                # 将文件大小从字节转换为MB
                file_size_mb = file_size / (4* 1024 * 1024)

                # 如果文件小于指定的最大大小，则复制
                if file_size_mb < max_size_mb:
                    # 构建目标文件路径
                    relative_path = os.path.relpath(src_file_path, src_folder)
                    dst_file_path = os.path.join(dst_folder, relative_path)

                    # 确保目标文件夹存在
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                    # 复制文件
                    shutil.copy2(src_file_path, dst_file_path)  # 使用copy2以保留元数据
                    if info:
                        print(f"copy: {src_file_path} -> {dst_file_path}")

# 构造最终的 JSON 输出
def to_json_with_ranklist(filtered_layers: List[Dict[str, int]]) -> str:
    # 构造一个字典，其中 'ranklist' 是包含所有层信息的列表
    output_dict = {
        "ranklist": filtered_layers
    }
    
    # 将字典转换为 JSON 字符串
    return json.dumps(output_dict, indent=4)



# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc, tokenizer


 

def merge_safe_tensor_save(sketch_file_path, merge_file_path, mtype = 'sketch'):
    sketch_1 = load_file(sketch_file_path)
    model_weights = load_file(merge_file_path)
    merge_data = {}
    
    if mtype == 'sketch':
        for layer_name, weights in model_weights.items():
            merge_data[layer_name] = weights
        for layer_name, weights in sketch_1.items():
            merge_data[layer_name] = weights

    elif mtype == 'origin':
        for layer_name, weights in model_weights.items():
            merge_data[layer_name] = weights
        for layer_name, weights in sketch_1.items():
            if layer_name.endswith('.weight') and weights.dim() > 1:
                merge_data[layer_name] = weights
    else:
        print("error merge type!!!!")
    f = safe_open(merge_file_path, framework="pt", device="cpu")
    metadata = f.metadata()
    save_file(merge_data, merge_file_path, metadata=metadata)


def update_config_file(file_path, new_json_str):
    # 读取现有的配置文件
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
    except FileNotFoundError:
        config = {}  # 如果文件不存在，则初始化为空字典

    # 解析新的 JSON 字符串
    try:
        new_config = json.loads(new_json_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return


    # 更新 config
    if 'quantization_config' in config:
        config['quantization_config'].update(new_config)
    else:
        config.update(new_config)


    # 写回更新后的配置
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    print("update config.")

def delete_tmp_files(path):
    try:
        shutil.rmtree(path)
        print(f"Successfully deleted folder: {path}")
    except FileNotFoundError:
        print(f"Folder not found: {path}")



def change_sketch_name(model_file_path):
    model_weights = load_file(model_file_path)
    model_weights_new = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for layer_name, weights in model_weights.items():
        if re.search(r'weight_R$', layer_name):
            # 在前面添加 'model.'
            layer_name = f"model.{layer_name}"
        if re.search(r'weight_L$', layer_name):
            # 在前面添加 'model.'
            layer_name = f"model.{layer_name}"
        # if re.search(r'weight$', layer_name):
        #     if layer_name.endswith('.weight') and weights.dim() > 1:
        #     # 在前面添加 'model.'
        #         layer_name = f"model.{layer_name}"
        # if layer_name == 'model.lm_head.weight' or layer_name == 'model.model.decoder.embed_positions.weight' or layer_name == 'model.model.decoder.embed_tokens.weight':
        #     layer_name = layer_name.replace("model.", "", 1)
        model_weights_new[layer_name] = weights
    f = safe_open(model_file_path, framework="pt", device="cpu")
    metadata = f.metadata()
    save_file(model_weights_new, model_file_path, metadata=metadata)    
    
def full_process(model_file_path, output_file_path, name = 'Llama-2-7b', qbit = 4, fix_rank = 0, ratio = 0.1, groupsize=128, info = False, isRTN = False):
    # first we change the module weight name 
    print("copy small files!")
    output_file_path_tmp = output_file_path
    copy_small_files(model_file_path, output_file_path_tmp,info = info)

    # 定义旧文件名和新文件名


    if name == 'Llama-2-7b':
        orgin_file_name1 = model_file_path + '/model-00001-of-00002.safetensors'
        orgin_file_name2 = model_file_path + '/model-00002-of-00002.safetensors'

        model_weights1 = load_file(orgin_file_name1)
        model_weights2 = load_file(orgin_file_name2)

        f = safe_open(orgin_file_name1, framework="pt", device="cpu")
        # 获取元数据
        metadata = f.metadata()

        combined_weights = merge_weight(model_weights1,model_weights2)
    elif name  == 'Llama-2-13b':
        orgin_file_name1 = model_file_path + '/model-00001-of-00003.safetensors'
        orgin_file_name2 = model_file_path + '/model-00002-of-00003.safetensors'
        orgin_file_name3 = model_file_path + '/model-00003-of-00003.safetensors'
        model_weights1 = load_file(orgin_file_name1)
        model_weights2 = load_file(orgin_file_name2)
        model_weights3 = load_file(orgin_file_name3)
        f = safe_open(orgin_file_name1, framework="pt", device="cpu")
        # 获取元数据
        metadata = f.metadata()
        combined_weights = merge_weight(model_weights1,model_weights2)    
        combined_weights = merge_weight(combined_weights,model_weights3)  
    elif name  == 'opt-125m' or name  == 'opt-350m' or name  == 'opt-1.3b' or name  == 'opt-2.7b' or  name  == 'opt-6.7b' or  name  == 'opt-13b':
        orgin_file_name1 = model_file_path + '/model.safetensors'

        model_weights1 = load_file(orgin_file_name1)

        f = safe_open(orgin_file_name1, framework="pt", device="cpu")
        # 获取元数据
        metadata = f.metadata()
        combined_weights = model_weights1   
    elif name  == 'bloom-560m' or name  == 'bloom-1b1' or name  == 'bloom-1b7'  or name  == 'bloom-7b1':
        orgin_file_name1 = model_file_path + '/model.safetensors'

        model_weights1 = load_file(orgin_file_name1)

        f = safe_open(orgin_file_name1, framework="pt", device="cpu")
        # 获取元数据
        metadata = f.metadata()
        combined_weights = model_weights1           
    print("load weight and combine!")

    if isRTN:
        print("now begin to rtn quant")
        with timing_context("RTN"):
            quant_RTN_save_full_process(combined_weights, output_file_path_tmp ,bit = qbit ,metadata = metadata, groupsize = groupsize)
    
    else:
        print("now begin to sketch")
        with timing_context("sketch"):
            quant_sketch_save_full_process(combined_weights, output_file_path_tmp, bit = qbit ,metadata = metadata,fix_rank = fix_rank, ratio = ratio, groupsize = groupsize, info = info)
    

        
    print("finish qiantize and save.")
    


    print("finish all process!")





if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', type=str,
        help='model path to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--output_path', type=str,
        help='fake quantize modle output path.'
    )
    parser.add_argument(
        '--model_name', type=str,
        help='modle name.'
    )
    parser.add_argument(
        '--qbit', type=int, default=4,
        help='bits to use for fake quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=128,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--lora_ratio', type=float, default=0.1,
        help='low rank approximate ratio.'
    )
    parser.add_argument(
        '--fix_rank', type=int, default=0,
        help='fix low rank. default 0 means do not fix.'
    )

    parser.add_argument(
        '--info', action='store_true',
        help='Whether to print sketch infos.'
    ) 
    parser.add_argument(
        '--isRTN', action='store_true',
        help='Whether to use RTN.'
    ) 
    args = parser.parse_args()

    full_process(
        args.model_path, args.output_path, name = args.model_name,
        qbit = args.qbit, fix_rank = args.fix_rank, ratio = args.lora_ratio,
        groupsize = args.groupsize, info = args.info, isRTN = args.isRTN,
    )

