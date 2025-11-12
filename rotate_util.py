from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import transformers
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
from hadamard_utils import *


from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import typing



OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
DEV = torch.device('cuda:0')

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_type_extractor(model)
    for W in get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    model.lm_head = model.lm_head.cuda()
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    model.lm_head = model.lm_head.cpu()



def rotate_attention_inputs(layer, Q, model_type, lora = None) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    dtype = layer.self_attn.q_proj.weight.dtype
    if lora is not None:
        if lora.get("self_attn.q_proj") is not None:
            lora["self_attn.q_proj"] = torch.matmul(lora["self_attn.q_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        if lora.get("self_attn.k_proj") is not None:
            lora["self_attn.k_proj"] = torch.matmul(lora["self_attn.k_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        if lora.get("self_attn.v_proj") is not None:
            lora["self_attn.v_proj"] = torch.matmul(lora["self_attn.v_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
                
def rotate_attention_inputs_test(layer, Q, model_type, lora = None) -> None:
    if lora is not None:
        dtype = lora["self_attn.q_proj"].dtype
        lora["self_attn.q_proj"] = torch.matmul(lora["self_attn.q_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        lora["self_attn.k_proj"] = torch.matmul(lora["self_attn.k_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        lora["self_attn.v_proj"] = torch.matmul(lora["self_attn.v_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
              

def rotate_attention_output(layer, Q, model_type, lora = None) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    
    if lora is not None:
        if lora.get("self_attn.o_proj") is not None:
            dtype = lora["self_attn.o_proj"].dtype
            lora["self_attn.o_proj"] = torch.matmul(Q.T.cuda(), lora["self_attn.o_proj"].double().cuda()).to(device="cpu", dtype=dtype)


def rotate_attention_output_test(layer, Q, model_type, lora = None) -> None:
    if lora is not None:
        dtype = lora["self_attn.o_proj"].dtype
        lora["self_attn.o_proj"] = torch.matmul(Q.T.cuda(), lora["self_attn.o_proj"].double().cuda()).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, model_type, lora = None):
    #print(Q)
    # Rotate the MLP input weights.
    if model_type == LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    dtype = layer.mlp.up_proj.weight.dtype

    if lora is not None:
        if lora.get("mlp.up_proj") is not None:
            lora["mlp.up_proj"] = torch.matmul(lora["mlp.up_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        if lora.get("mlp.gate_proj") is not None:
            lora["mlp.gate_proj"] = torch.matmul(lora["mlp.gate_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)



def rotate_mlp_input_test(layer, Q, model_type, lora = None):
    if lora is not None:
        dtype = lora["mlp.gate_proj"].dtype
        lora["mlp.up_proj"] = torch.matmul(lora["mlp.up_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        lora["mlp.gate_proj"] = torch.matmul(lora["mlp.gate_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        
    
def rotate_mlp_output(layer, Q, model_type, lora = None):
    # Rotate the MLP output weights and bias.
    if model_type == LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    
    if lora is not None:
        if lora.get("mlp.down_proj") is not None:
            dtype = lora["mlp.down_proj"].dtype
            lora["mlp.down_proj"] = torch.matmul(Q.T.cuda(), lora["mlp.down_proj"].double().cuda()).to(device="cpu", dtype=dtype)

def rotate_mlp_output_h(layer, Q, model_type, lora = None):
    # Rotate the MLP output weights and bias.
    if model_type == LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    
    if lora is not None:
        if lora.get("mlp.down_proj") is not None:
            dtype = lora["mlp.down_proj"].dtype
            lora["mlp.down_proj"] = torch.matmul(Q.T.cuda(), lora["mlp.down_proj"].double().cuda()).to(device="cpu", dtype=dtype)
            lora["mlp.down_proj"] = apply_exact_had_to_linear(W, had_dim=-1, output=False, lora = lora["mlp.down_proj"])
        else:
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
    else:
        apply_exact_had_to_linear(W, had_dim=-1, output=False)


def rotate_mlp_output_test(layer, Q, model_type, lora = None):
    if lora is not None:
        dtype = lora["mlp.down_proj"].dtype
        lora["mlp.down_proj"] = torch.matmul(Q.T.cuda(), lora["mlp.down_proj"].double().cuda()).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, model_type, head_num, head_dim, lora = None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    if lora is not None:
        if lora.get("self_attn.v_proj") is not None: 
            lora["self_attn.v_proj"] = apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, lora = lora["self_attn.v_proj"])
        else:
            apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)

        if lora.get("self_attn.o_proj") is not None: 
            lora["self_attn.o_proj"] = apply_exact_had_to_linear(o_proj, had_dim=-1, output=False, lora = lora["self_attn.o_proj"])
        else:
            apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    else:
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def rotate_head_no_fuse_norm(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    model.lm_head = model.lm_head.cuda()
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    model.lm_head = model.lm_head.cpu()
    #model.model.norm.weight.data = torch.matmul(model.model.norm.weight.to(device=DEV, dtype=torch.float64), Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs_no_fuse_norm(layer, Q, model_type, lora = None) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    if lora is not None:
        dtype = lora["self_attn.q_proj"].dtype
        lora["self_attn.q_proj"] = torch.matmul(lora["self_attn.q_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        lora["self_attn.k_proj"] = torch.matmul(lora["self_attn.k_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        lora["self_attn.v_proj"] = torch.matmul(lora["self_attn.v_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
    #layer.input_layernorm.weight.data = torch.matmul(layer.input_layernorm.weight.to(device=DEV, dtype=torch.float64) , Q).to(device="cpu", dtype=dtype)

def rotate_mlp_input_no_fuse_norm(layer, Q, model_type, lora = None):
    # Rotate the MLP input weights.
    if model_type == LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    if lora is not None:
        dtype = lora["mlp.gate_proj"].dtype
        lora["mlp.up_proj"] = torch.matmul(lora["mlp.up_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
        lora["mlp.gate_proj"] = torch.matmul(lora["mlp.gate_proj"].double().cuda(), Q.cuda()).to(device="cpu", dtype=dtype)
    #layer.post_attention_layernorm.weight.data = torch.matmul(layer.post_attention_layernorm.weight.to(device=DEV, dtype=torch.float64), Q).to(device="cpu", dtype=dtype)
    

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        #print(W_.shape, layernorm.weight.shape)
        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            
def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type

def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)

def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def fuse_layer_norms(model):
    
    model_type = get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
    
        if model_type == OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(get_pre_head_layernorm(**kwargs), [get_lm_head(**kwargs)])
    
    replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    

def fuse_layer_norms2(model, WR):
    
    model_type = get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    i=0
    for layer in layers:
        if i<=2:
            dtype = WR[i]["mlp.gate_proj"].dtype
            WR[i]["self_attn.q_proj"]= (WR[i]["self_attn.q_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)
            WR[i]["self_attn.k_proj"]= (WR[i]["self_attn.k_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)
            WR[i]["self_attn.v_proj"]= (WR[i]["self_attn.v_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)

            WR[i]["mlp.gate_proj"]= (WR[i]["mlp.gate_proj"].double().cpu() * layer.post_attention_layernorm.weight.double()).to(dtype)
            WR[i]["mlp.up_proj"]= (WR[i]["mlp.up_proj"].double().cpu() * layer.post_attention_layernorm.weight.double()).to(dtype)

            i = i+1
        # fuse the input layernorms into the linear layers
        if model_type == LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
    
        if model_type == OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(get_pre_head_layernorm(**kwargs), [get_lm_head(**kwargs)])
    
    replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: RMSN(model.config.hidden_size),
        replace_layers=False,
    )


def fuse_layer_norms2_retNorm(model, WR):
    
    model_type = get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    i=0

    layer_norm = []
    for layer in layers:
        ln = {}
        if i<=31:
            dtype = WR[i]["mlp.gate_proj"].dtype
            WR[i]["self_attn.q_proj"]= (WR[i]["self_attn.q_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)
            WR[i]["self_attn.k_proj"]= (WR[i]["self_attn.k_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)
            WR[i]["self_attn.v_proj"]= (WR[i]["self_attn.v_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)
            ln["input_layernorm"] = layer.input_layernorm.weight
            WR[i]["mlp.gate_proj"]= (WR[i]["mlp.gate_proj"].double().cpu() * layer.post_attention_layernorm.weight.double()).to(dtype)
            WR[i]["mlp.up_proj"]= (WR[i]["mlp.up_proj"].double().cpu() * layer.post_attention_layernorm.weight.double()).to(dtype)
            ln["post_attention_layernorm"] = layer.post_attention_layernorm.weight
            layer_norm.append(ln)
            i = i+1
        # fuse the input layernorms into the linear layers
        if model_type == LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    

    fuse_ln_linear(get_pre_head_layernorm(**kwargs), [get_lm_head(**kwargs)])
    
    replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    return layer_norm

def fuse_layer_norms2_test(model, WR):
    
    model_type = get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    

    layers = get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    i=0
    for layer in layers:
        if i<=31:
            dtype = WR[i]["mlp.gate_proj"].dtype
            WR[i]["self_attn.q_proj"]= (WR[i]["self_attn.q_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)
            WR[i]["self_attn.k_proj"]= (WR[i]["self_attn.k_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)
            WR[i]["self_attn.v_proj"]= (WR[i]["self_attn.v_proj"].double().cpu() * layer.input_layernorm.weight.double()).to(dtype)

            WR[i]["mlp.gate_proj"]= (WR[i]["mlp.gate_proj"].double().cpu() * layer.post_attention_layernorm.weight.double()).to(dtype)
            WR[i]["mlp.up_proj"]= (WR[i]["mlp.up_proj"].double().cpu() * layer.post_attention_layernorm.weight.double()).to(dtype)

            i = i+1



def model_layer_rotate(model, layers):

    fuse_layer_norms(model)
    # torch.manual_seed(42)
    # random_matrix = torch.randn(layers[0].self_attn.q_proj.weight.shape[0], layers[0].self_attn.q_proj.weight.shape[0], dtype=torch.float64)


    # Q, R = torch.qr(random_matrix, some=True)
    # Q *= torch.sign(torch.diag(R)).unsqueeze(0)
    # Q = Q.cuda()
    # #print(Q)
    # Q = Q.to(torch.float64)
    Q = random_hadamard_matrix(layers[0].self_attn.q_proj.weight.shape[0], DEV)# block_diagonal_walsh_matrix(layers[0].self_attn.q_proj.weight.shape[0],128,DEV)

    for i in tqdm.tqdm(range(len(layers)), desc="Rotating Model and fuse norm..."):
        model_type = model_type_extractor(model)
        rotate_attention_inputs(layers[i], Q, model_type)
        rotate_attention_output(layers[i], Q, model_type)
        rotate_mlp_input(layers[i], Q, model_type)
        rotate_mlp_output(layers[i], Q, model_type)        


    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    return Q




def model_layer_rotate2(model, layers, WR):

    #print(WR[0]["self_attn.q_proj"])
    fuse_layer_norms2(model, WR)


    # random_matrix = torch.randn(layers[0].self_attn.q_proj.weight.shape[0], layers[0].self_attn.q_proj.weight.shape[0], dtype=torch.float64)


    # Q, R = torch.qr(random_matrix, some=True)
    # Q *= torch.sign(torch.diag(R)).unsqueeze(0)
    # Q = Q.cuda()
    
    # Q = Q.to(torch.float64)
    Q = random_hadamard_matrix(layers[0].self_attn.q_proj.weight.shape[0], DEV)
    for i in tqdm.tqdm(range(len(layers)), desc="Rotating Model and fuse norm..."):
        model_type = model_type_extractor(model)
        if i<=2:
            lora = WR[i]
        else:
            lora = None
        rotate_attention_inputs(layers[i], Q, model_type, lora)
        rotate_attention_output(layers[i], Q, model_type, lora)
        rotate_mlp_input(layers[i], Q, model_type, lora)
        rotate_mlp_output(layers[i], Q, model_type, lora)        

    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    return Q, WR



def model_layer_rotate2_test(model, layers, WR):

    res = []
    i=0
    for layer in layers:
        r = {}
        if i<=3:
            dtype = WR[i]["mlp.gate_proj"].dtype
            r["self_attn.q_proj"]= layer.self_attn.q_proj.weight.double().cpu() - WR[i]["self_attn.q_proj"].double().cpu()
            r["self_attn.k_proj"]= layer.self_attn.k_proj.weight.double().cpu() - WR[i]["self_attn.k_proj"].double().cpu()
            r["self_attn.v_proj"]= layer.self_attn.v_proj.weight.double().cpu() - WR[i]["self_attn.v_proj"].double().cpu()
            r["self_attn.o_proj"]= layer.self_attn.o_proj.weight.double().cpu() - WR[i]["self_attn.o_proj"].double().cpu()

            r["mlp.gate_proj"]= layer.mlp.gate_proj.weight.double().cpu()  - WR[i]["mlp.gate_proj"].double().cpu()
            r["mlp.up_proj"]= layer.mlp.up_proj.weight.double().cpu()  - WR[i]["mlp.up_proj"].double().cpu()
            r["mlp.down_proj"]= layer.mlp.down_proj.weight.double().cpu()  - WR[i]["mlp.down_proj"].double().cpu()
     
            res.append(r)
            i = i+1
    #print(WR[0]["self_attn.q_proj"])

    print("-1",res[0]["self_attn.q_proj"])
    # fuse_layer_norms2_test(model, res)
    # fuse_layer_norms2(model, WR)
    
 
    Q = random_hadamard_matrix(layers[0].self_attn.q_proj.weight.shape[0], DEV)
    for i in tqdm.tqdm(range(len(layers)), desc="Rotating Model and fuse norm test..."):
        model_type = model_type_extractor(model)
        if i<=3:
            lora = WR[i]
            lora2 = res[i]
        else:
            lora = None
            lora2 = None
        rotate_attention_inputs(layers[i], Q, model_type, lora)
        rotate_attention_output(layers[i], Q, model_type, lora)
        rotate_mlp_input(layers[i], Q, model_type, lora)
        rotate_mlp_output(layers[i], Q, model_type, lora)        

        rotate_attention_inputs_test(layers[i], Q, model_type, lora2)
        rotate_attention_output_test(layers[i], Q, model_type, lora2)
        rotate_mlp_input_test(layers[i], Q, model_type, lora2)
        rotate_mlp_output_test(layers[i], Q, model_type, lora2) 

    print("0",WR[0]["self_attn.q_proj"].double().cpu())
    print("1",res[0]["self_attn.q_proj"])
    print("2",layers[0].self_attn.q_proj.weight.double().cpu())
    print("3",res[0]["self_attn.q_proj"] -  layers[0].self_attn.q_proj.weight.double().cpu() +  WR[0]["self_attn.q_proj"].double().cpu())
    i=0
    for layer in layers:
        if i<=3:
            dtype = WR[i]["mlp.gate_proj"].dtype
            print(torch.norm(res[i]["self_attn.q_proj"] -  layer.self_attn.q_proj.weight.double().cpu() +  WR[i]["self_attn.q_proj"].double().cpu())/torch.norm(layer.self_attn.q_proj.weight))
            print(torch.norm(res[i]["self_attn.k_proj"] -  layer.self_attn.k_proj.weight.double().cpu() +  WR[i]["self_attn.k_proj"].double().cpu())/torch.norm(layer.self_attn.k_proj.weight))
            print(torch.norm(res[i]["self_attn.v_proj"] -  layer.self_attn.v_proj.weight.double().cpu() +  WR[i]["self_attn.v_proj"].double().cpu())/torch.norm(layer.self_attn.v_proj.weight))
            print(torch.norm(res[i]["self_attn.o_proj"] -  layer.self_attn.o_proj.weight.double().cpu() +  WR[i]["self_attn.o_proj"].double().cpu())/torch.norm(layer.self_attn.o_proj.weight))
            print(torch.norm(res[i]["mlp.gate_proj"] -  layer.mlp.gate_proj.weight.double().cpu() +  WR[i]["mlp.gate_proj"].double().cpu())/torch.norm(layer.mlp.gate_proj.weight))
            print(torch.norm(res[i]["mlp.up_proj"] -  layer.mlp.up_proj.weight.double().cpu() +  WR[i]["mlp.up_proj"].double().cpu())/torch.norm(layer.mlp.up_proj.weight))
            print(torch.norm(res[i]["mlp.down_proj"] -  layer.mlp.down_proj.weight.double().cpu() +  WR[i]["mlp.down_proj"].double().cpu())/torch.norm(layer.mlp.down_proj.weight))
            i = i+1
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    return Q, WR



def model_layer_rotate2_norotate(model, layers, WR):
    Q = random_hadamard_matrix(layers[0].self_attn.q_proj.weight.shape[0], DEV)
    # import numpy as np
    # Q = walsh_matrix(int(np.log2(layers[0].self_attn.q_proj.weight.shape[0])), DEV)
    #Q3 = block_diagonal_walsh_matrix(layers[0].self_attn.q_proj.weight.shape[0],2, DEV)

    #Q = Q3
    for i in tqdm.tqdm(range(len(layers)), desc="Rotating Model and fuse norm..."):
        model_type = model_type_extractor(model)
        if i<=2:
            lora = WR[i]
        else:
            lora = None
        rotate_attention_inputs(layers[i], Q, model_type, lora)
        rotate_attention_output(layers[i], Q, model_type, lora)
        rotate_mlp_input(layers[i], Q, model_type, lora)
        rotate_mlp_output(layers[i], Q, model_type, lora)        

    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    return Q, WR


def model_layer_rotate2_norotate_h(model, layers, WR):
    #Q = random_hadamard_matrix(layers[0].self_attn.q_proj.weight.shape[0], DEV)
    # import numpy as np
    # Q = walsh_matrix(int(np.log2(layers[0].self_attn.q_proj.weight.shape[0])), DEV)
    Q = block_diagonal_walsh_matrix(layers[0].self_attn.q_proj.weight.shape[0],128,DEV)

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    for i in tqdm.tqdm(range(len(layers)), desc="Rotating Model and fuse norm..."):
        model_type = model_type_extractor(model)
        if i<=31:
            lora = WR[i]
        else:
            lora = None
        rotate_attention_inputs(layers[i], Q, model_type, lora)
        rotate_attention_output(layers[i], Q, model_type, lora)
        rotate_mlp_input(layers[i], Q, model_type, lora)
        rotate_mlp_output_h(layers[i], Q, model_type, lora)

        rotate_ov_proj(layers[i], model_type, num_heads, head_dim, lora)
 
    rotate_embeddings(model, Q)
    rotate_head(model, Q)

    quant_utils.add_actquant(model) #Add Activation Wrapper to the model
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if 'down_proj' in name:
            had_K, K = get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = False
        if 'o_proj' in name:
            had_K, K = get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            qlayers[name].fp32_had = False
    return Q, WR



def model_layer_rotate2_retNorm(model, layers, WR):

    #print(WR[0]["self_attn.q_proj"])
    layer_norm = fuse_layer_norms2_retNorm(model, WR)


    # random_matrix = torch.randn(layers[0].self_attn.q_proj.weight.shape[0], layers[0].self_attn.q_proj.weight.shape[0], dtype=torch.float64)


    # Q, R = torch.qr(random_matrix, some=True)
    # Q *= torch.sign(torch.diag(R)).unsqueeze(0)
    # Q = Q.cuda()
    
    # Q = Q.to(torch.float64)
    Q = random_hadamard_matrix(layers[0].self_attn.q_proj.weight.shape[0], DEV)
    for i in tqdm.tqdm(range(len(layers)), desc="Rotating Model and fuse norm..."):
        model_type = model_type_extractor(model)
        if i<=31:
            lora = WR[i]
        else:
            lora = None
        rotate_attention_inputs(layers[i], Q, model_type, lora)
        rotate_attention_output(layers[i], Q, model_type, lora)
        rotate_mlp_input(layers[i], Q, model_type, lora)
        rotate_mlp_output(layers[i], Q, model_type, lora)        

    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    return Q, WR, layer_norm

