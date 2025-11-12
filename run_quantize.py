from get_calib_data import get_wikitext2, get_pile, get_wikitext2_, get_redPajamas, get_c4_
from get_scale_quant import quant_sketch, scale_quant_layer, scale_quant_layer_save,count_mem, scale_quant_layer_ret_WR, scale_quant_only, _quant_only, scale_quant_only_double, scale_quant_only_rnorm,scale_quant_layer_ret_WR_2
from eval_util import *
from rotate_util import *
import gptq_utils

import gptq_utils_lora
import gptvq_utils
import gptvq_utils_lora
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import torch
import torch.nn as nn
import argparse
import os
import json

import tqdm
import gc
import functools
from collections import defaultdict
from typing import List
from utils import copy_small_files

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# from tinychat.models import LlavaLlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

#from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from quant_utils import *

qtype = torch.float16

def get_blocks(model):
    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM","Qwen3ForCausalLM"):
        layers = model.model.layers
    # elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
    #     # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
    #     layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    #if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM)):
    
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    # elif isinstance(model, LlavaLlamaForCausalLM):
    #     model.model.embed_tokens = model.model.embed_tokens.to(device)
    #     model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif "llavallamamodel" in str(model.__class__).lower():
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_named_linears_wp(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, ActQuantWrapper)}

@torch.no_grad()
def run_model_quant(model_path, output_path, qbit = 4, fix_rank = 0, ratio = 0.1, group_size = 128, info = False, save_lora = False):    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    kwargs = {"torch_dtype": qtype, "low_cpu_mem_usage": True}

    copy_small_files(model_path, output_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs
    )
    model.eval()

    # traindataset, testenc, tokenizer = get_wikitext2(256, 0, 2048, pretrained_model_dir)

    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    samples = get_pile(
        data="wikitext2", tokenizer=enc, n_samples=128, block_size=512
    )
    samples = torch.cat(samples, dim=0)


    # samples, testenc, tokenizer = get_wikitext2(256, 0, 2048, model_path)
    print(samples.shape)
    layers = get_blocks(model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()


    quant_infos = {}
    quant_infos["lora_rank"] = 0.0
    quant_infos["lora_size"] = 0.0
    quant_infos["total_size"] = 0.0
    quant_infos["quant_size"] = 0.0
    quant_infos["layer_cnt"] = 0.0
    quant_infos["quant_size"] = 0.0
    quant_infos["origin_size"] = 0.0
    loras_dict = {}

    #print(layers[0].mlp.gate_proj.weight)
    #Q = model_layer_rotate(model, layers)
    #print(layers[0].mlp.gate_proj.weight)
    for i in tqdm.tqdm(range(len(layers)), desc="Running FLRQ..."):

        # if i>0:
        #     break
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        # # get output as next layer's input
        inps_org = inps
        inps  = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()


        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        if save_lora:
            s_key = "model.layers."+str(i)
            layer,lora = scale_quant_layer_save(layer, input_feat,quant_infos ,w_bit=qbit, fix_rank=fix_rank, ratio = ratio, group_size = group_size)
            
            for key,val in lora.items():
                ke = s_key+"."+key
                loras_dict[ke] = val
        else:
            layer = scale_quant_layer(layer, input_feat,quant_infos ,w_bit=qbit, fix_rank=fix_rank, ratio = ratio, group_size = group_size, index = i)


        #inps  = layer(inps_org, **layer_kwargs)[0]
        #count_mem(layer, input_feat,quant_infos ,w_bit=qbit, fix_rank=fix_rank, ratio = ratio, group_size = group_size)
        layers[i] = layer.cpu()
        layer = layer.cpu()

        del input_feat,handles
        gc.collect()
        torch.cuda.empty_cache()


    

    dataloader, testloader, _ = get_wikitext2(128, 0, 2048, '/rjs/ghyx/data/Llama-2-7b')
    llama_eval(model, testloader, DEV)


    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        model.model.layers = nn.ModuleList(layers)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = nn.ModuleList(layers)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = nn.ModuleList(layers)


    state_dict = model.state_dict()

    q_bit = 16.0*(float(quant_infos["quant_size"])/float(quant_infos["total_size"]))
    loraq_bit = 16.0*(float(quant_infos["quant_size"]+quant_infos["lora_size"])/float(quant_infos["total_size"]))
    avg_rank = quant_infos["lora_rank"]/quant_infos["layer_cnt"]
    print(f"avg_rank = {avg_rank}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    # org_size = quant_infos["origin_size"]/(1000.0*1000.0)
    # qsize = quant_infos["quant_size"]/(1000.0*1000.0)
    # print(f"mem_org = {org_size}, qmem(group) = {qsize}")


    model.save_pretrained(output_path)
    enc.save_pretrained(output_path)

    if save_lora:
        save_file(loras_dict, output_path+"/lora.safetensors", metadata={"format": "pt"})



@torch.no_grad()
def run_model_quant2(model_path, output_path, qbit = 4, fix_rank = 0, ratio = 0.1, group_size = 128, info = False, save_lora = False):    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    kwargs = {"torch_dtype": qtype, "low_cpu_mem_usage": True}

    copy_small_files(model_path, output_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs
    )
    model.eval()

    # traindataset, testenc, tokenizer = get_wikitext2(256, 0, 2048, pretrained_model_dir)

    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    samples = get_pile(
        data="wikitext2", tokenizer=enc, n_samples=128, block_size=512
    )
    samples = torch.cat(samples, dim=0)


    # samples, testenc, tokenizer = get_wikitext2(256, 0, 2048, model_path)
    #print(samples.shape)
    layers = get_blocks(model)
    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    #del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()


    quant_infos = {}
    quant_infos["lora_rank"] = 0.0
    quant_infos["lora_size"] = 0.0
    quant_infos["total_size"] = 0.0
    quant_infos["quant_size"] = 0.0
    quant_infos["layer_cnt"] = 0.0
    quant_infos["quant_size"] = 0.0
    quant_infos["origin_size"] = 0.0
    loras_dict = {}

    WR = []
    fuse_layer_norms(model)
    # #print(layers[0].self_attn.q_proj.weight)
    for i in tqdm.tqdm(range(len(layers)), desc="Running FLRQ..."):

        if i>31:
            break
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        # # get output as next layer's input
        inps_org = inps
        inps  = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()


        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        

        layer, W_R_layer = scale_quant_layer_ret_WR(layer, input_feat,quant_infos ,w_bit=qbit, fix_rank=fix_rank, ratio = ratio, group_size = group_size, index = i)
        WR.append(W_R_layer)
        # layer = scale_quant_layer(layer, input_feat,quant_infos ,w_bit=qbit, fix_rank=fix_rank, ratio = ratio, group_size = group_size, index = i)
              

        layers[i] = layer.cpu()
        layer = layer.cpu()

        del input_feat,handles
        gc.collect()
        torch.cuda.empty_cache()
    

    model_layer_rotate2_norotate(model, layers, WR)
    # model_layer_rotate(model, layers)
 
    # #model_layer_rotate2_test(model, layers, WR)
    # #Q, WR, layer_norm = model_layer_rotate2_retNorm(model, layers, WR)
    # #model_layer_rotate2(model, layers)

    layers = get_blocks(model)
    inps = []
    layer_kwargs = {}
    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    model = model.to("cuda")
    samples = samples.to("cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")



    gc.collect()
    torch.cuda.empty_cache()

    print(len(layers))
    for i in tqdm.tqdm(range(len(layers)), desc="Running FLRQ2..."):
        if i>31:
            break
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps_org = inps
        inps  = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
  
        # print(named_linears, inps)

        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}


        #layer = scale_quant_only_rnorm(layer, WR[i], input_feat, layer_norm[i],w_bit=qbit, group_size = 128, index = i)   
        layer = scale_quant_only(layer, WR[i], input_feat, w_bit=qbit, group_size = 128, index = i)
        #layer = _quant_only(layer, input_feat, w_bit=qbit, group_size = group_size)


        layers[i] = layer.cpu()
        layer = layer.cpu()

        del input_feat,handles
        gc.collect()
        torch.cuda.empty_cache()

    del WR
    gc.collect()

    dataloader, testloader, _ = get_wikitext2(128, 0, 2048, '/rjs/ghyx/data/Llama-2-7b')
    llama_eval(model, testloader, DEV)


    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        model.model.layers = nn.ModuleList(layers)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = nn.ModuleList(layers)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = nn.ModuleList(layers)


    state_dict = model.state_dict()

    q_bit = 16.0*(float(quant_infos["quant_size"])/float(quant_infos["total_size"]))
    loraq_bit = 16.0*(float(quant_infos["quant_size"]+quant_infos["lora_size"])/float(quant_infos["total_size"]))
    avg_rank = quant_infos["lora_rank"]/quant_infos["layer_cnt"]
    print(f"avg_rank = {avg_rank}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    # org_size = quant_infos["origin_size"]/(1000.0*1000.0)
    # qsize = quant_infos["quant_size"]/(1000.0*1000.0)
    # print(f"mem_org = {org_size}, qmem(group) = {qsize}")


    # model.save_pretrained(output_path)
    # enc.save_pretrained(output_path)

    if save_lora:
        save_file(loras_dict, output_path+"/lora.safetensors", metadata={"format": "pt"})



@torch.no_grad()
def run_model_quant3(model_path, output_path, qbit = 4, fix_rank = 0, ratio = 0.1, group_size = 128, info = False, save_lora = False, method = "gptq", lora_bit = 8,loratool="sketch", ha_bsize = 256, id_bsize = 256,lora_iter = 8):    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    kwargs = {"torch_dtype": qtype, "low_cpu_mem_usage": True}

    copy_small_files(model_path, output_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs
    )
    model.eval()

    # traindataset, testenc, tokenizer = get_wikitext2(256, 0, 2048, pretrained_model_dir)

    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    #red_pajama = get_redPajamas(128, 0, 4096, '/rjs/ghyx/data/Llama-2-7b')

    
    samples = get_pile(
        data="c4", tokenizer=enc, n_samples=128, block_size=4096
    )
    samples = torch.cat(samples, dim=0)


    # samples, testenc, tokenizer = get_wikitext2(256, 0, 2048, model_path)
    #print(samples.shape)
    layers = get_blocks(model)
    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]




    quant_infos = {}
    quant_infos["lora_rank"] = 0.0
    quant_infos["lora_size"] = 0.0
    quant_infos["total_size"] = 0.0
    quant_infos["quant_size"] = 0.0
    quant_infos["layer_cnt"] = 0.0
    quant_infos["quant_size"] = 0.0
    quant_infos["origin_size"] = 0.0
    loras_dict = {}

    WR = []
    #fuse_layer_norms(model)
    #model_layer_rotate(model, layers)

    for i in tqdm.tqdm(range(len(layers)), desc="Running FLRQ..."):

        if i>100:
            break
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        # # get output as next layer's input
        inps_org = inps
        inps  = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()


        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        

        layer, W_R_layer = scale_quant_layer_ret_WR_2(layer, input_feat,quant_infos ,w_bit=qbit, fix_rank=fix_rank, ratio = ratio, group_size = group_size, index = i, lora_bit = lora_bit,loratool=loratool,lora_iter = lora_iter)
        WR.append(W_R_layer)
        #layer = scale_quant_layer(layer, input_feat,quant_infos ,w_bit=qbit, fix_rank=fix_rank, ratio = ratio, group_size = group_size, index = i)
                    
        layers[i] = layer.cpu()
        layer = layer.cpu()

        del input_feat,handles
        gc.collect()
        torch.cuda.empty_cache()
    
    del inps
    #model_layer_rotate2(model, layers, WR)
    #model_layer_rotate2_norotate(model, layers, WR)
    #model_layer_rotate(model, layers)

    from argparse import Namespace

    args = Namespace()
    args.nsamples = 128
    args.w_bits = qbit
    args.w_asym = True
    args.int8_down_proj = False
    args.w_clip = True
    args.w_groupsize = group_size
    args.percdamp = 0.01
    args.act_order = False
    args.id_bsize = id_bsize
    args.ha_bsize = ha_bsize
    args.act_order = False
    model.seqlen = 4096
    dataloader, testloader, _ = get_wikitext2_(128, 0, model.seqlen, model_path)

    if method == "gptq":
        quantizers = gptq_utils_lora.gptq_fwrd_lora(model, WR, dataloader, DEV, args)
    else:
        quantizers = gptvq_utils_lora.gptvq_fwrd_lora(model, WR, dataloader, DEV, args)

    llama_eval(model, testloader, DEV)


    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        model.model.layers = nn.ModuleList(layers)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = nn.ModuleList(layers)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = nn.ModuleList(layers)


    state_dict = model.state_dict()


    model.save_pretrained(output_path)
    enc.save_pretrained(output_path)

    
    q_bit = 16.0*(float(quant_infos["quant_size"])/float(quant_infos["total_size"]))
    loraq_bit = 16.0*(float(quant_infos["quant_size"]+quant_infos["lora_size"])/float(quant_infos["total_size"]))
    avg_rank = quant_infos["lora_rank"]/quant_infos["layer_cnt"]
    print(f"avg_rank = {avg_rank}, qbit(group) = {q_bit},  loraq-bit(group) = {loraq_bit},")
    org_size = quant_infos["origin_size"]/(1000.0*1000.0)
    qsize = quant_infos["quant_size"]/(1000.0*1000.0)
    print(f"mem_org = {org_size}, qmem(group) = {qsize}")


    if save_lora:
        save_file(loras_dict, output_path+"/lora.safetensors", metadata={"format": "pt"})


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
        '--lora_iter', type=int, default=8,
        help='the iteration of r1-sketch.'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Whether to print sketch infos.'
    ) 
    parser.add_argument(
        '--save_lora', action='store_true',
        help='Whether to save loras.'
    ) 

    parser.add_argument(
        '--lora_bit', type=int, default=16,
        help='The bit of U,V in low-rank.'
    ) 
    parser.add_argument(
        '--method', type=str, default="gptq",
        help='quantize method.'
    )
    parser.add_argument(
        '--loratool', type=str, default="sketch",
        help='low-rank method.'
    )

    parser.add_argument(
        '--ha_bsize', type=int, default=256,
        help='hadamard blocksize.'
    )


    parser.add_argument(
        '--id_bsize', type=int, default=256,
        help='Identity blocksize.'
    )
    args = parser.parse_args()

    run_model_quant3(
        args.model_path, args.output_path, qbit = args.qbit, 
        fix_rank = args.fix_rank, ratio = args.lora_ratio,
        group_size = args.groupsize, info = args.info, save_lora=args.save_lora, 
        method = args.method, lora_bit = args.lora_bit, loratool = args.loratool,
        ha_bsize = args.ha_bsize, id_bsize = args.id_bsize, lora_iter = args.lora_iter
    )

    # run_model_quant(
    #     args.model_path, args.output_path, qbit = args.qbit, 
    #     fix_rank = args.fix_rank, ratio = args.lora_ratio,
    #     group_size = args.groupsize, info = args.info, save_lora=args.save_lora, 
    # )