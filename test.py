from get_calib_data import get_wikitext2, get_pile, get_wikitext2_, get_redPajamas, get_c4_
from get_scale_quant import quant_sketch, scale_quant_layer, scale_quant_layer_save,count_mem, scale_quant_layer_ret_WR, scale_quant_only, _quant_only, scale_quant_only_double, scale_quant_only_rnorm
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
from quantizer import *
qtype = torch.float16


import numpy as np
import pandas as pd

def tensor_to_excel(tensor, filename, sheet_name='Sheet1'):
    """
    将二维tensor（PyTorch或NumPy）保存到Excel文件的指定工作表中。

    参数:
    tensor: 二维的 PyTorch Tensor 或 NumPy 数组
    filename: str, 输出的Excel文件名（如 'output.xlsx'）
    sheet_name: str, Excel中的工作表名称（默认为 'Sheet1'）
    """
    # 检查输入是否为二维
    # if len(tensor.shape) != 2:
    #     raise ValueError(f"输入的tensor必须是二维的，当前shape为 {tensor.shape}")

    # 如果是 PyTorch tensor，转换为 NumPy 数组
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()  # 确保在CPU上并转为numpy
    elif isinstance(tensor, np.ndarray):
        array = tensor
    else:
        raise TypeError("输入必须是 PyTorch Tensor 或 NumPy 数组")

    # 使用 pandas 将数组转换为 DataFrame
    df = pd.DataFrame(array)

    # 写入 Excel 文件
    try:
        with pd.ExcelWriter(filename, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        print(f"成功将tensor保存到 '{filename}' 的 '{sheet_name}' 工作表中。")
    except FileNotFoundError:
        # 如果文件不存在，创建新文件
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        print(f"成功创建 '{filename}' 并保存tensor到 '{sheet_name}' 工作表中。")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == '__main__':
    import argparse
    mean = 0.0
    std = 0.5
    W = torch.randn(8, 8) * std + mean
    torch.set_printoptions(
        precision=7,        # 保留7位小数
        sci_mode=False      # 禁用科学计数法
    )

    tensor_to_excel(W, "algo.xlsx", "W_org")
    X = [0.2, 2.0, 0.5, 0.4, 5.0, 0.1, 0.4, 0.6]
    
    partial_size = 2
    rotate_size = 2
    W = W.cuda()
    X = torch.tensor(X).cuda()

    tensor_to_excel(X, "algo.xlsx", "scale")
    lora,res,L,R = sketch_pre_Test(W, X, fix_rank = 2, bit = 4)

    tensor_to_excel(L, "algo.xlsx", "W_L")
    tensor_to_excel(R, "algo.xlsx", "W_R")
    tensor_to_excel(res, "algo.xlsx", "res_org")
    abs_tensor = res.abs()
    mean_abs_per_col = abs_tensor.mean(dim=0) 
    #top_values, top_indices = torch.topk(-mean_abs_per_col, W.shape[1])

    perm = mean_abs_per_col
    top_values, top_indices = torch.topk(-perm, W.shape[1])

    
    print(top_indices)

    rot = block_diagonal_walsh_matrix(W.shape[1], rotate_size, W.device)
    diagI_rot = create_diagI_matrix_upper(rot, rot.shape[0]-partial_size).to(W.device).to(W.dtype)
    Ppermute = construct_partial_permutation_matrix_upper(top_indices, m = W.shape[1], dtype = W.dtype).to(W.device)
    PD_rot = Ppermute @ diagI_rot
    tensor_to_excel(Ppermute, "algo.xlsx", "Ppermute")
    tensor_to_excel(diagI_rot, "algo.xlsx", "diag_Hadamard")
    print(diagI_rot)
    res = res.to(Ppermute.dtype) @ Ppermute
    print(res)
    res = res.to(diagI_rot.dtype) @ diagI_rot
    print(res)
    tensor_to_excel(res, "algo.xlsx", "res_rotate")
    res = pseudo_quantize_tensor(res, bit=2,q_group_size=4)

    print(res)
    tensor_to_excel(res, "algo.xlsx", "res_rotate_quant")
