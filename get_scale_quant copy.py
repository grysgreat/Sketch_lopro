import torch
import torch.nn as nn
import argparse
import os
import json

from quantizer import *
from utils import *
from get_clip_weight import *
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer
# from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RMSNorm


SCALE_CLAMP_MIN = 1e-4


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)

@torch.no_grad()
def quant_sketch(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

    qweight,srank = quant_sketch_save_full_process_pre(layer.weight, scales, input_feat, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight


@torch.no_grad()
def quant_sketch_clip(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight)
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=20, max_shrink=0.5, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
    if (srank!=0):
        qweight = q_res + lora_W.T
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight
    
@torch.no_grad()
def quant_sketch_clip_iter(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight)
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=20, max_shrink=0.9, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
    
    w_res = layer.weight - q_res
    lora_W,srank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight)    
    
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.5, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
    

    if (srank!=0):
        qweight = q_res + lora_W.T
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight
    

@torch.no_grad()
def quant_sketch_clip_iter_for(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight)
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)


    iter = 10
    max_shrinks = [0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5]
    
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=max_shrinks[i], n_sample_token=512)
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
        

    if (srank!=0):
        qweight = q_res + lora_W.T
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight
    
@torch.no_grad()
def quant_sketch_clip_iter_for2(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T

    #test
    q_res = pseudo_quantize_tensor(layer.weight, bit=w_bit,q_group_size=group_size)
    scale_q = input_feat @ (q_res.T)
    loss_q = (scale_org.to(input_feat.device) - scale_q.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_q = loss_q/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    #####

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    #test
    q_res = pseudo_quantize_tensor(res, bit=w_bit,q_group_size=group_size)
    scale_loraq = input_feat @ (q_res.T + lora_W)
    loss_loraq = (scale_org.to(input_feat.device) - scale_loraq.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_loraq = loss_loraq/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    ####
    if(rela_loss_loraq>0.18):
        w_bit = 3
    # if(rela_loss_loraq<0.1):
    #     w_bit = 1
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

    iter = 0


    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 20
    # elif w_bit == 1: 
    #     iter = 20    
    max_shrinks = [0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5]

    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=max_shrinks[i], n_sample_token=512)
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

        #print(q_res.T.shape, lora_W.shape)
        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
        #print(rela_loss,rela_loss_loraq)
        if loss < best_error:
            best_rela_loss = rela_loss
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res

    print(f"best_rela_loss: {best_rela_loss:.5f}, rela_loss_loraq: {rela_loss_loraq:.5f}, rela_loss_q: {rela_loss_q:.5f}, w_bit={w_bit}")
    print(best_rank)
    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight


@torch.no_grad()
def quant_sketch_clip_iter_for2_diff(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T

    #test
    q_res = pseudo_quantize_tensor(layer.weight, bit=w_bit,q_group_size=group_size)
    scale_q = input_feat @ (q_res.T)
    loss_q = (scale_org.to(input_feat.device) - scale_q.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_q = loss_q/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    #####

    lora_W,srank = sketch_pre_diff(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    #test
    q_res = pseudo_quantize_tensor(res, bit=w_bit,q_group_size=group_size)
    scale_loraq = input_feat @ (q_res.T + lora_W)
    loss_loraq = (scale_org.to(input_feat.device) - scale_loraq.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_loraq = loss_loraq/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    ####
    # if(rela_loss_loraq>0.18):
    #     w_bit = 3
    # if(rela_loss_loraq<0.1):
    #     w_bit = 1
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

    iter = 0


    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 20
    # elif w_bit == 1: 
    #     iter = 20    
    max_shrinks = [0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5]

    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = sketch_pre_diff(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=max_shrinks[i], n_sample_token=512)
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

        #print(q_res.T.shape, lora_W.shape)
        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
        #print(rela_loss,rela_loss_loraq)
        if loss < best_error:
            best_rela_loss = rela_loss
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res

    print(f"best_rela_loss: {best_rela_loss:.5f}, rela_loss_loraq: {rela_loss_loraq:.5f}, rela_loss_q: {rela_loss_q:.5f}, w_bit={w_bit}")
    print(best_rank)
    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight
    
@torch.no_grad()
def quant_sketch_clip_iter_for3(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.5):


    #print(layer.weight.shape, input_feat.shape)
    fix_rank = 32
    max_clip = 0.5

    #判断奇异值分布或者均值，然后再选clip，奇异值接近，rank不高的，clip取低
    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    
    
    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    #mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.pow(2.4)

    #mean_feat  = mean_feat/mean_feat.mean()
    if mean_feat.dtype==torch.float16:
        mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    if mean_feat.dtype==torch.bfloat16:
        mean_feat = mean_feat.clamp(min=1e-14)
    
    

    # if(mean_feat.amax()/mean_feat.amin() > 1e6):
    #     max_clip = 0.3
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    
    
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T


    #test
    q_res = pseudo_quantize_tensor(layer.weight, bit=w_bit,q_group_size=group_size)
    scale_q = input_feat @ (q_res.T)
    loss_q = (scale_org.to(input_feat.device) - scale_q.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_q = loss_q/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    #####

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    #print((lora_W.T-layer.weight).amax(),layer.weight.amax())
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    
    #print(res.shape,input_feat.shape)
    clip_res,kl0 = auto_clip_lora2_test(layer.weight, res, lora_W, input_feat, w_bit, group_size=group_size, n_grid=20, max_shrink=max_clip, n_sample_token=512)
    #clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor_do(clip_res, bit=w_bit,q_group_size=group_size)
    #print(res-clip_res)

    scale_loraq = input_feat @ (q_res.T + lora_W)
    loss_loraq = (scale_org.to(input_feat.device) - scale_loraq.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss0 = loss_loraq/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    
    print("rela_loss0:",rela_loss0, "; KL0:",kl0)
    iter = 0
    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 10
    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None

    
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        clip_res,kl = auto_clip_lora2_test(layer.weight, res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=max_clip, n_sample_token=512)
        
        #clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=0.5, n_sample_token=512)
        q_res = pseudo_quantize_tensor_do(clip_res, bit=w_bit,q_group_size=group_size)
        #print(q_res.T.shape, lora_W.shape)
        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
        
        loss = (kl0-kl)/kl0 + (rela_loss0-rela_loss)/rela_loss0
        print(kl,rela_loss, loss)

        #print(input_feat.amin(),input_feat.amax(),input_feat.shape)
        if loss < best_error:
            best_rela_loss = rela_loss
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res
    

    # w_res = layer.weight - best_qres
    # best_lora,best_rank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    # best_lora = best_lora.T

    print("best_error:",best_error)
    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res
    
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight



    
@torch.no_grad()
def quant_sketch_clip_iter_for3_down_o(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.5):

    fix_rank = 64
    max_clip = 0.15
    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    mean_feat = mean_feat.pow(2.4)

    if mean_feat.dtype==torch.float16:
        mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    if mean_feat.dtype==torch.bfloat16:
        mean_feat = mean_feat.clamp(min=1e-14)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T


    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)

    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    

    #40,0.5 , 20,0.3: 6.69
    clip_res,kl = auto_clip_lora2_test(layer.weight, res, lora_W, input_feat, w_bit, group_size=group_size, n_grid=40, max_shrink=max_clip, n_sample_token=512)
    # if kl > 0.5:
    #     w_bit = 3

    #clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor_do(clip_res, bit=w_bit,q_group_size=group_size)


    iter = 0
    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 20
    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None

    
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        clip_res,kl = auto_clip_lora2_test(layer.weight, res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=max_clip, n_sample_token=512)
        # if kl > 0.5:
        #     w_bit = 3
        q_res = pseudo_quantize_tensor_do(clip_res, bit=w_bit,q_group_size=group_size)

        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())

        # print(rela_loss)
        # if loss < best_error:
        #     best_rela_loss = rela_loss
        #     best_error = loss
        #     best_iter = i
        #     best_rank = srank
        #     best_lora = lora_W.T         
        #     best_qres = q_res
        best_rela_loss = rela_loss
        best_error = loss
        best_iter = i
        best_rank = srank
        best_lora = lora_W.T         
        best_qres = q_res    


    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res
    print("best:",best_rela_loss, w_bit)
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight



    
@torch.no_grad()
def quant_sketch_clip_iter_for3_lora_fp8(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T

    #test
    q_res = pseudo_quantize_tensor(layer.weight, bit=w_bit,q_group_size=group_size)
    scale_q = input_feat @ (q_res.T)
    loss_q = (scale_org.to(input_feat.device) - scale_q.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_q = loss_q/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    #####
    # lora_W,srank = sketch_pre_print_mse(layer.weight, scales,input_feat, fix_rank = 128, bit = w_bit, ratio = ratio, groupsize = group_size)
    

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    #test
    q_res = pseudo_quantize_tensor(res, bit=w_bit,q_group_size=group_size)
    scale_loraq = input_feat @ (q_res.T + lora_W)
    loss_loraq = (scale_org.to(input_feat.device) - scale_loraq.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_loraq = loss_loraq/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    ####
    # if(rela_loss_loraq>0.18):
    #     w_bit = 3
    # if(rela_loss_loraq<0.05):
    #     w_bit = 2
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

    iter = 0


    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 20

    max_shrinks = [0.8,0.4,0.5,0.5,0.5,0.7,0.8,0.9,0.9,0.9,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.9,0.8,0.9]
    max_shrinks2 = [0.5,0.5,0.5,0.5]
    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None
    best_lora_L = None
    best_lora_R = None
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank,r1_L,r1_R = sketch_pre_full_ret(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=0.5, n_sample_token=512)
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
        # name = "fig-"+str(i)
        # draw_hist(res[0,0:128],name)
        # name = "fig-clip-"+str(i)
        # draw_hist(clip_res[0,0:128],name)

        #print(q_res.T.shape, lora_W.shape)
        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
        #print(rela_loss,rela_loss_loraq)
        if loss < best_error:
            best_rela_loss = rela_loss
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res
            best_lora_L = r1_L
            best_lora_R = r1_R

    # res = layer.weight - best_qres
    # lora_W,srank = sketch_pre_mse(res,best_qres,layer.weight, scales,input_feat, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)

    # best_lora_L = best_lora_L.to(torch.float8_e5m2)
    # best_lora_L = best_lora_L.to(torch.float16)

    # best_lora_R = best_lora_R.to(torch.float8_e4m3fn)
    # best_lora_R = best_lora_R.to(torch.float16)

    if (best_rank!=0):
        lora_W = torch.matmul(best_lora_L.T,best_lora_R)
        lora_W = torch.diag(scales.cuda().float()).inverse().half() @ lora_W
        best_lora = lora_W.T

    # best_lora = best_lora.to(torch.float8_e5m2)
    # best_lora = best_lora.to(torch.float16)
    # print(f"best_rela_loss: {best_rela_loss:.5f}, rela_loss_loraq: {rela_loss_loraq:.5f}, rela_loss_q: {rela_loss_q:.5f}, w_bit={w_bit}")
    # print(best_rank)
    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight


@torch.no_grad()
def quant_sketch_clip_iter_for3_lora_fp8_ret(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T

    #test
    q_res = pseudo_quantize_tensor(layer.weight, bit=w_bit,q_group_size=group_size)
    scale_q = input_feat @ (q_res.T)
    loss_q = (scale_org.to(input_feat.device) - scale_q.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_q = loss_q/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    #####
    # lora_W,srank = sketch_pre_print_mse(layer.weight, scales,input_feat, fix_rank = 128, bit = w_bit, ratio = ratio, groupsize = group_size)
    

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    #test
    q_res = pseudo_quantize_tensor(res, bit=w_bit,q_group_size=group_size)
    scale_loraq = input_feat @ (q_res.T + lora_W)
    loss_loraq = (scale_org.to(input_feat.device) - scale_loraq.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_loraq = loss_loraq/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    ####
    # if(rela_loss_loraq>0.18):
    #     w_bit = 3
    # if(rela_loss_loraq<0.05):
    #     w_bit = 2
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

    iter = 0


    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 20

    max_shrinks = [0.8,0.4,0.5,0.5,0.5,0.7,0.8,0.9,0.9,0.9,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.9,0.8,0.9]
    max_shrinks2 = [0.5,0.5,0.5,0.5]
    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None
    best_lora_L = None
    best_lora_R = None
    best_S = None
    for i in range(iter):
        w_res = layer.weight - q_res
        #lora_W,srank0 = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        #lora_W,srank1 = sketch_pre_fp8(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
                
        lora_W,srank,r1_L,r1_R,S_arr = sketch_pre_fp8_ret(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        #print(srank0, srank1, srank)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=0.5, n_sample_token=512)
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
        # name = "fig-"+str(i)
        # draw_hist(res[0,0:128],name)
        # name = "fig-clip-"+str(i)
        # draw_hist(clip_res[0,0:128],name)

        #print(q_res.T.shape, lora_W.shape)
        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
        #print(rela_loss,rela_loss_loraq)
        if loss < best_error:
            best_rela_loss = rela_loss
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res
            best_lora_L = r1_L
            best_lora_R = r1_R
            best_S = S_arr
    print(best_rank)
    # w_res = layer.weight - best_qres
    # best_lora,srank,best_lora_L,best_lora_R,best_S = sketch_pre_fp8_ret(w_res, scales, fix_rank = best_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    # res = layer.weight - best_lora.T
    # clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=0.5, n_sample_token=512)
    # best_qres = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)


    # res = layer.weight - best_qres
    # lora_W,srank = sketch_pre_mse(res,best_qres,layer.weight, scales,input_feat, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)

    # best_lora_L = best_lora_L.to(torch.float8_e5m2)
    # best_lora_L = best_lora_L.to(torch.float16)

    # best_lora_R = best_lora_R.to(torch.float8_e4m3fn)
    # best_lora_R = best_lora_R.to(torch.float16)

    if (best_rank!=0):
        #print(best_lora_L[0].dtype)
        best_lora_L = [tensor.to(torch.float16) for tensor in best_lora_L]
        best_lora_R = [tensor.to(torch.float16) for tensor in best_lora_R]


        best_lora_L = torch.vstack(best_lora_L[:best_rank])
        best_lora_R = torch.vstack(best_lora_R[:best_rank])
        best_lora_L = best_lora_L.T @  torch.diag(torch.tensor(best_S).cuda().half())
        best_lora_L = best_lora_L.T


        lora_W = torch.matmul(best_lora_L.T,best_lora_R)
        lora_W = torch.diag(scales.cuda().float()).inverse().half() @ lora_W
        best_lora = lora_W.T

    # best_lora = best_lora.to(torch.float8_e5m2)
    # best_lora = best_lora.to(torch.float16)
    # print(f"best_rela_loss: {best_rela_loss:.5f}, rela_loss_loraq: {rela_loss_loraq:.5f}, rela_loss_q: {rela_loss_q:.5f}, w_bit={w_bit}")
    # print(best_rank)
    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight



@torch.no_grad()
def quant_sketch_clip_iter_for3_print_mse(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos, name = "default"):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T


    # lora_W,srank = sketch_pre_print_mse(layer.weight, scales,input_feat, fix_rank = 128, bit = w_bit, ratio = ratio, groupsize = group_size, name = name)
    lora_W,srank, absval, mse = sketch_pre_mse_ret(layer.weight, scales,input_feat, fix_rank = 128, bit = w_bit, ratio = ratio, groupsize = group_size, name = name)

    data = {
        'absval': absval,
        'mse': mse
    }
    df = pd.DataFrame(data)
    file_name = "abs_mse_llama2-7b.xlsx"
    # 将 DataFrame 写入 Excel 文件，指定工作表名称
    # index=False 表示不将 DataFrame 的索引写入 Excel
    try:
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            df.to_excel(writer, sheet_name=name, index=False)
        print(f"数据已成功追加到 '{file_name}' 的新工作表 '{name}' 中。")
    except FileNotFoundError:
        # 如果文件不存在，就创建它（使用默认的 'w' 模式）
        df.to_excel(file_name, sheet_name=name, index=False)
        print(f"文件 '{file_name}' 不存在，已创建并写入工作表 '{name}'。")
    except ValueError as e:
        if "Sheet name" in str(e) and "already exists" in str(e):
            print(f"错误：工作表名称 '{name}' 在文件 '{file_name}' 中已存在。请使用不同的名称。")
        else:
            raise e


    if (srank!=0):
        qweight = lora_W
    else:
        qweight = lora_W
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight

    
@torch.no_grad()
def quant_sketch_clip_iter_for3_save(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T

    #test
    q_res = pseudo_quantize_tensor(layer.weight, bit=w_bit,q_group_size=group_size)
    scale_q = input_feat @ (q_res.T)
    loss_q = (scale_org.to(input_feat.device) - scale_q.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_q = loss_q/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    #####
    # lora_W,srank = sketch_pre_print_mse(layer.weight, scales,input_feat, fix_rank = 128, bit = w_bit, ratio = ratio, groupsize = group_size)
    

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    #test
    q_res = pseudo_quantize_tensor(res, bit=w_bit,q_group_size=group_size)
    scale_loraq = input_feat @ (q_res.T + lora_W)
    loss_loraq = (scale_org.to(input_feat.device) - scale_loraq.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_loraq = loss_loraq/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    ####
    # if(rela_loss_loraq>0.18):
    #     w_bit = 3
    # if(rela_loss_loraq<0.05):
    #     w_bit = 2
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

    iter = 0


    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 20

    max_shrinks = [0.8,0.4,0.5,0.5,0.5,0.7,0.8,0.9,0.9,0.9,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.9,0.8,0.9]
    max_shrinks2 = [0.5,0.5,0.5,0.5]
    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None
    best_lora_L = None
    best_lora_R = None
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=0.5, n_sample_token=512)
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
        # name = "fig-"+str(i)
        # draw_hist(res[0,0:128],name)
        # name = "fig-clip-"+str(i)
        # draw_hist(clip_res[0,0:128],name)

        #print(q_res.T.shape, lora_W.shape)
        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
        #print(rela_loss,rela_loss_loraq)
        if loss < best_error:
            best_rela_loss = rela_loss
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res

    # res = layer.weight - best_qres
    # lora_W,srank = sketch_pre_mse(res,best_qres,layer.weight, scales,input_feat, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)


    # print(f"best_rela_loss: {best_rela_loss:.5f}, rela_loss_loraq: {rela_loss_loraq:.5f}, rela_loss_q: {rela_loss_q:.5f}, w_bit={w_bit}")
    # print(best_rank)
    if (best_rank!=0):
        qweight = best_qres + best_lora
        best_lora_L = best_lora[:,0:best_rank].contiguous()
        best_lora_R = best_lora[0:best_rank,:].contiguous()
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight,best_lora_L,best_lora_R,best_rela_loss

@torch.no_grad()
def quant_sketch_clip_iter_for3_ablation_calib(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight)
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=20, max_shrink=0.5, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
    if (srank!=0):
        qweight = q_res + lora_W.T
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight
    
@torch.no_grad()
def quant_sketch_clip_iter_for3_ununi(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)

    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T

    #test
    q_res = pseudo_quantize_tensor_2bit(layer.weight, bit=w_bit,q_group_size=group_size)
    scale_q = input_feat @ (q_res.T)
    loss_q = (scale_org.to(input_feat.device) - scale_q.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_q = loss_q/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    #####

    lora_W,srank = sketch_pre(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight.T)
    #test
    q_res = pseudo_quantize_tensor_2bit(res, bit=w_bit,q_group_size=group_size)
    scale_loraq = input_feat @ (q_res.T + lora_W)
    loss_loraq = (scale_org.to(input_feat.device) - scale_loraq.to(input_feat.device)).float().pow(2).sum().item()
    rela_loss_loraq = loss_loraq/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
    ####
    if(rela_loss_loraq>0.18):
        w_bit = 3
    if(rela_loss_loraq<0.05):
        w_bit = 2
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    if w_bit == 3:
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
    else:
        q_res = pseudo_quantize_tensor_2bit(clip_res, bit=w_bit,q_group_size=group_size)
    iter = 0


    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 1
    elif w_bit == 2: 
        iter = 20

    max_shrinks = [0.8,0.4,0.5,0.5,0.5,0.7,0.8,0.9,0.9,0.9,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.9,0.8,0.9]

    best_rela_loss = 0.0
    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = sketch_pre(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora_ununi(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=0.5, n_sample_token=512)
        if w_bit == 2:
            q_res = pseudo_quantize_tensor_2bit(clip_res, bit=w_bit,q_group_size=group_size)
        else:
            q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
        # name = "fig-"+str(i)
        # draw_hist(res[0,0:128],name)
        # name = "fig-clip-"+str(i)
        # draw_hist(clip_res[0,0:128],name)
        #print(q_res.T.shape, lora_W.shape)
        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        rela_loss = loss/ (scale_org.to(input_feat.device).float().pow(2).sum().item())
        #print(rela_loss,rela_loss_loraq)
        if loss < best_error:
            best_rela_loss = rela_loss
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res

    # res = layer.weight - best_qres
    # lora_W,srank = sketch_pre_mse(w_res,best_qres,layer.weight, scales,input_feat, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)


    print(f"best_rela_loss: {best_rela_loss:.5f}, rela_loss_loraq: {rela_loss_loraq:.5f}, rela_loss_q: {rela_loss_q:.5f}, w_bit={w_bit}")
    print(best_rank)
    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res
    #print(qweight)
    #print(mean_feat.max(), mean_feat.min(),scales.max(), scales.min())
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight




@torch.no_grad()
def count_memory(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):
    zero_avg = (0.001+w_bit)/128.0
    quant_infos["origin_size"] =  quant_infos["origin_size"] + layer.weight.element_size() * layer.weight.nelement()
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.element_size() * layer.weight.nelement() * ((w_bit+0.125+zero_avg)/16.0)
    qweight = 0
    return qweight


@torch.no_grad()
def scale_quant_layer(layer, input_feat, quant_infos, w_bit=4, fix_rank = 0, ratio = 0.1, group_size = 128, index = 1):
    # fix_rank = 11
    # print(layer)
    if isinstance(layer, OPTDecoderLayer):
        layer.self_attn.q_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.k_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.v_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.out_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attn.out_proj , input_feat["self_attn.out_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.fc1.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.fc1 , input_feat["fc1"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.fc2.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.fc2 , input_feat["fc2"] , w_bit, group_size, fix_rank, ratio, quant_infos))
    #elif isinstance(layer, (LlamaDecoderLayer, Qwen2DecoderLayer, Qwen3DecoderLayer)):
    elif isinstance(layer, (LlamaDecoderLayer, Qwen2DecoderLayer)):
      # attention input
      #ppl is very high at llama2 2b quantization, o_proj and down_proj can't do clip.
        if index<31:
            layer.self_attn.q_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.3))
            layer.self_attn.k_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.3))
            layer.self_attn.v_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.3))
            layer.self_attn.o_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_down_o(layer.self_attn.o_proj , input_feat["self_attn.o_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.3))      
            layer.mlp.gate_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.mlp.gate_proj , input_feat["mlp.gate_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.3))
            layer.mlp.down_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_down_o(layer.mlp.down_proj , input_feat["mlp.down_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.3))
            layer.mlp.up_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.mlp.up_proj , input_feat["mlp.up_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.3))
    
        # if (index)%8 == 0 or index==31:
        #     layer.self_attn.q_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_print_mse(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos, name = str(index)+"_q_proj"))
        #     layer.self_attn.k_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_print_mse(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos, name = str(index)+"_k_proj"))
        #     layer.self_attn.v_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_print_mse(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos, name = str(index)+"_v_proj"))
        #     layer.self_attn.o_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_print_mse(layer.self_attn.o_proj , input_feat["self_attn.o_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos, name = str(index)+"_o_proj"))      
        #     layer.mlp.gate_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_print_mse(layer.mlp.gate_proj , input_feat["mlp.gate_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos, name = str(index)+"_gate_proj"))
        #     layer.mlp.down_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_print_mse(layer.mlp.down_proj , input_feat["mlp.down_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos, name = str(index)+"_down_proj"))
        #     layer.mlp.up_proj.weight = nn.Parameter(quant_sketch_clip_iter_for3_print_mse(layer.mlp.up_proj , input_feat["mlp.up_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos, name = str(index)+"_up_proj"))
        

    
    elif isinstance(layer, BloomBlock):
        layer.self_attention.query_key_value.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attention.query_key_value , input_feat["self_attention.query_key_value"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attention.dense.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.self_attention.dense , input_feat["self_attention.dense"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.mlp.dense_h_to_4h.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.mlp.dense_h_to_4h , input_feat["mlp.dense_h_to_4h"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.mlp.dense_4h_to_h.weight = nn.Parameter(quant_sketch_clip_iter_for3(layer.mlp.dense_4h_to_h , input_feat["mlp.dense_4h_to_h"] , w_bit, group_size, fix_rank, ratio, quant_infos))
    return layer



@torch.no_grad()
def scale_quant_layer_save(layer, input_feat, quant_infos, w_bit=4, fix_rank = 0, ratio = 0.1, group_size = 128):
    # fix_rank = 11
    lora_dict = {}
    if isinstance(layer, OPTDecoderLayer):
        q_proj,q_lora_L,q_lora_R = quant_sketch_clip_iter_for3_save(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        k_proj,k_lora_L,k_lora_R = quant_sketch_clip_iter_for3_save(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        v_proj,v_lora_L,v_lora_R = quant_sketch_clip_iter_for3_save(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        o_proj,o_lora_L,o_lora_R = quant_sketch_clip_iter_for3_save(layer.self_attn.out_proj , input_feat["self_attn.out_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        fc1,fc1_lora_L,fc1_lora_R = quant_sketch_clip_iter_for3_save(layer.fc1 , input_feat["fc1"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        fc2,fc2_lora_L,fc2_lora_R = quant_sketch_clip_iter_for3_save(layer.fc2 , input_feat["fc2"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        layer.self_attn.q_proj.weight = nn.Parameter(q_proj)
        layer.self_attn.k_proj.weight = nn.Parameter(k_proj)
        layer.self_attn.v_proj.weight = nn.Parameter(v_proj)
        layer.self_attn.out_proj.weight = nn.Parameter(o_proj)
        layer.fc1.weight = nn.Parameter(fc1)
        layer.fc2.weight = nn.Parameter(fc2)
        if q_lora_L!= None:
            lora_dict["q_proj.l"] = q_lora_L
            lora_dict["q_proj.r"] = q_lora_R
        if k_lora_L!= None:
            lora_dict["k_proj.l"] = k_lora_L
            lora_dict["k_proj.r"] = k_lora_R
        if v_lora_L!= None:
            lora_dict["v_proj.l"] = v_lora_L
            lora_dict["v_proj.r"] = v_lora_R
        if o_lora_L!= None:
            lora_dict["o_proj.l"] = o_lora_L
            lora_dict["o_proj.r"] = o_lora_R
        if fc1_lora_L!= None:
            lora_dict["fc1.l"] = fc1_lora_L
            lora_dict["fc1.r"] = fc1_lora_R
        if fc2_lora_L!= None:
            lora_dict["fc2.l"] = fc2_lora_L 
            lora_dict["fc2.r"] = fc2_lora_R
    elif isinstance(layer, (LlamaDecoderLayer, Qwen2DecoderLayer)):
        q_proj,q_lora_L,q_lora_R,rela_loss_q  = quant_sketch_clip_iter_for3_save(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        k_proj,k_lora_L,k_lora_R,rela_loss_k  = quant_sketch_clip_iter_for3_save(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        v_proj,v_lora_L,v_lora_R,rela_loss_v  = quant_sketch_clip_iter_for3_save(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        o_proj,o_lora_L,o_lora_R,rela_loss_o  = quant_sketch_clip_iter_for3_save(layer.self_attn.o_proj , input_feat["self_attn.o_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        gate_proj,gate_proj_lora_L,gate_proj_lora_R,rela_loss_gate  = quant_sketch_clip_iter_for3_save(layer.mlp.gate_proj , input_feat["mlp.gate_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        down_proj,down_proj_lora_L,down_proj_lora_R,rela_loss_down  = quant_sketch_clip_iter_for3_save(layer.mlp.down_proj , input_feat["mlp.down_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        up_proj,up_proj_lora_L,up_proj_lora_R,rela_loss_up = quant_sketch_clip_iter_for3_save(layer.mlp.up_proj , input_feat["mlp.up_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        layer.self_attn.q_proj.weight = nn.Parameter(q_proj)
        layer.self_attn.k_proj.weight = nn.Parameter(k_proj)
        layer.self_attn.v_proj.weight = nn.Parameter(v_proj)
        layer.mlp.gate_proj.weight = nn.Parameter(gate_proj)
        layer.mlp.down_proj.weight = nn.Parameter(down_proj)
        layer.self_attn.o_proj.weight = nn.Parameter(o_proj)
        layer.mlp.up_proj.weight = nn.Parameter(up_proj)

        print(f"rloss_q: {rela_loss_q:.3f}, rloss_k: {rela_loss_k:.3f}, rloss_v: {rela_loss_v:.3f}, rloss_o: {rela_loss_o:.3f}, rloss_gate: {rela_loss_gate:.5f}, rloss_down: {rela_loss_down:.3f}, rloss_up: {rela_loss_up:.3f}")

        if q_lora_L!= None:
            lora_dict["q_proj.l"] = q_lora_L
            lora_dict["q_proj.r"] = q_lora_R
        if k_lora_L!= None:
            lora_dict["k_proj.l"] = k_lora_L
            lora_dict["k_proj.r"] = k_lora_R
        if v_lora_L!= None:
            lora_dict["v_proj.l"] = v_lora_L
            lora_dict["v_proj.r"] = v_lora_R
        if o_lora_L!= None:
            lora_dict["o_proj.l"] = o_lora_L
            lora_dict["o_proj.r"] = o_lora_R
        if gate_proj_lora_L!= None:
            lora_dict["gate_proj.l"] = gate_proj_lora_L
            lora_dict["gate_proj.r"] = gate_proj_lora_R
        if down_proj_lora_L!= None:
            lora_dict["down_proj.l"] = down_proj_lora_L
            lora_dict["down_proj.r"] = down_proj_lora_R
        if up_proj_lora_L!= None:
            lora_dict["up_proj.l"] = up_proj_lora_L
            lora_dict["up_proj.r"] = up_proj_lora_R


    return layer, lora_dict







@torch.no_grad()
def count_mem(layer, input_feat, quant_infos, w_bit=4, fix_rank = 0, ratio = 0.1, group_size = 128):
    # fix_rank = 11
    lora_dict = {}
    if isinstance(layer, OPTDecoderLayer):
        count_memory(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.self_attn.out_proj , input_feat["self_attn.out_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.fc1 , input_feat["fc1"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.fc2 , input_feat["fc2"] , w_bit, group_size, fix_rank, ratio, quant_infos)

    elif isinstance(layer, (LlamaDecoderLayer, Qwen2DecoderLayer)):
        count_memory(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.self_attn.o_proj , input_feat["self_attn.o_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.mlp.gate_proj , input_feat["mlp.gate_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.mlp.down_proj , input_feat["mlp.down_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)
        count_memory(layer.mlp.up_proj , input_feat["mlp.up_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos)

    return layer, lora_dict


