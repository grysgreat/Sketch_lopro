import torch
import torch.nn as nn
import gc
from quantizer import *
import torch.nn.functional as F

def calculate_kl_divergence_loss(W, W2, num_bins=100, eps=1e-7):
    """
    计算两个 PyTorch 权重矩阵之间的 KL 散度损失。
    
    Args:
        W (torch.Tensor): 原始权重矩阵。
        W2 (torch.Tensor): 量化后的权重矩阵。
        num_bins (int): 用于构建直方图的分桶数量。
        eps (float): 用于数值稳定的小值，防止除零或 log(0)。
    
    Returns:
        float: KL 散度损失值 (D_KL(P || Q))。
    """
    values_W = W.flatten()
    values_W2 = W2.flatten()

    min_val = min(values_W.min().item(), values_W2.min().item())
    max_val = max(values_W.max().item(), values_W2.max().item())
    
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=W.device)
    indices_W = torch.bucketize(values_W, bin_edges, right=True) - 1 
    indices_W2 = torch.bucketize(values_W2, bin_edges, right=True) - 1

    indices_W = torch.clamp(indices_W, 0, num_bins - 1)
    indices_W2 = torch.clamp(indices_W2, 0, num_bins - 1)
    

    hist_W = torch.bincount(indices_W, minlength=num_bins).float()
    hist_W2 = torch.bincount(indices_W2, minlength=num_bins).float()

    total_W = hist_W.sum() + eps
    total_W2 = hist_W2.sum() + eps
    P = hist_W / total_W  # 原始权重的分布
    Q = hist_W2 / total_W2 # 量化权重的分布

    log_ratio = torch.log((P + eps) / (Q + eps)) # 加 eps 保证数值稳定
    kl_div = torch.sum(P * log_ratio)
    
    return kl_div.item() # 返回标量值


import torch

def calculate_kl_divergence_loss_batch(W, W2, num_bins=100, eps=1e-7):
    """
    计算两个PyTorch张量最后一维度之间的KL散度损失（批处理版本）。
    
    Args:
        W (torch.Tensor): 原始权重张量，形状为[..., D]。
        W2 (torch.Tensor): 量化后的权重张量，形状与W相同。
        num_bins (int): 直方图的分桶数量。
        eps (float): 数值稳定的小值。
    
    Returns:
        torch.Tensor: KL散度损失，形状为[..., 1]。
    """
    # 保存原始形状（除最后一维）
    original_shape = W.shape[:-1]
    D = W.shape[-1]
    
    # 展平除最后一维外的所有维度
    W_flat = W.reshape(-1, D)
    W2_flat = W2.reshape(-1, D)
    N = W_flat.shape[0]  # 批处理大小
    
    # 计算全局最小值和最大值
    min_val = min(W_flat.min().item(), W2_flat.min().item())
    max_val = max(W_flat.max().item(), W2_flat.max().item())
    
    # 创建分桶边界
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=W.device)
    
    # 将值分配到桶中
    indices_W = torch.clamp(torch.bucketize(W_flat, bin_edges, right=True) - 1, 0, num_bins - 1)
    indices_W2 = torch.clamp(torch.bucketize(W2_flat, bin_edges, right=True) - 1, 0, num_bins - 1)
    
    # 初始化直方图
    hist_W = torch.zeros((N, num_bins), device=W.device)
    hist_W2 = torch.zeros((N, num_bins), device=W.device)
    
    # 使用散射填充直方图
    row_indices = torch.arange(N, device=W.device)[:, None].expand_as(indices_W)
    hist_W.scatter_add_(1, indices_W, torch.ones_like(indices_W, dtype=torch.float))
    hist_W2.scatter_add_(1, indices_W2, torch.ones_like(indices_W2, dtype=torch.float))
    
    # 计算概率分布
    P = hist_W / (hist_W.sum(dim=1, keepdim=True) + eps)
    Q = hist_W2 / (hist_W2.sum(dim=1, keepdim=True) + eps)
    
    # 计算KL散度
    log_ratio = torch.log((P + eps) / (Q + eps))
    kl_div = torch.sum(P * log_ratio, dim=1).to(W.dtype)
    
    # 重塑为原始形状（除最后一维外）并添加尾维
    return kl_div.reshape(original_shape + (1,))

def pseudo_quantize_tensor2(
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
    scales = (max_val - min_val).clamp(min=1e-16) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)


    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0


    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w

@torch.no_grad()
def auto_clip_lora2(
    W_O, w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    # w = w.to(torch.float64)
    # lora_W = lora_W.to(torch.float64)
    # input_feat = input_feat.to(torch.float64)

    lora_W = lora_W.T
    lora_W = lora_W.contiguous()
    w = w.contiguous()
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)
    # W_O = W_O.reshape(W_O.shape[0], 1, -1, group_size)
    lora_W = lora_W.reshape(lora_W.shape[0], 1, -1, group_size)
    oc_batch_size = 128 if w.shape[0] % 128 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, bit=n_bit,q_group_size=group_size)

            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del q_w
            del cur_out
            cur_best_idx = err < min_errs
            #print(cur_best_idx)
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        del org_w
        best_max_val_all.append(best_max_val)
        # print(bset_i)
    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    del input_feat
    gc.collect()
    torch.cuda.empty_cache()
    del org_out
    gc.collect()
    torch.cuda.empty_cache()


    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    #print(w_all.shape, best_max_val.shape)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)

    w_all = w_all.reshape(org_w_shape)

    return w_all

@torch.no_grad()
def auto_clip_lora(
    w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    W_O = lora_W.T + w
    lw_ori= lora_W.T
    #print(input_feat.shape)
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = group_size if group_size > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)
    lora_W = lora_W.reshape(w.shape[0], 1, -1, group_size)


    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, bit=n_bit,q_group_size=group_size)
            #q_w = pseudo_quantize_tensor0(cur_w, bit=n_bit,group_size=group_size)
            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()

    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    #print(w_all.shape, best_max_val.shape)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)
    w_all = w_all.reshape(org_w_shape)

    q_w_all = pseudo_quantize_tensor_do(w_all, bit=n_bit,q_group_size=group_size)
    lqw = q_w_all+lw_ori
    kl_loss = calculate_kl_divergence_loss(W_O, lqw, num_bins=200)
    #print(f"KL Divergence Loss 3: {kl_loss:.6f}")

    return w_all


@torch.no_grad()
def auto_clip_lora_org(
    w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    lora_W = lora_W.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, bit=n_bit,q_group_size=group_size)
            #q_w = pseudo_quantize_tensor0(cur_w, bit=n_bit,group_size=group_size)
            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    #print(w_all.shape, best_max_val.shape)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)

    w_all = w_all.reshape(org_w_shape)
    return w_all

@torch.no_grad()
def auto_clip_lora_test(
    w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = group_size if group_size > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)
    lora_W = lora_W.reshape(w.shape[0], 1, -1, group_size)


    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, bit=n_bit,q_group_size=group_size)
            #q_w = pseudo_quantize_tensor0(cur_w, bit=n_bit,group_size=group_size)
            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()

    w_all = w_all.reshape(*best_max_val.shape[:2], -1)

    w_all = torch.clamp(w_all, -best_max_val, best_max_val)

    w_all = w_all.reshape(org_w_shape)
    return w_all


@torch.no_grad()
def auto_clip_lora_ununi(
    w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    lora_W = lora_W.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            if n_bit == 2:
                q_w = pseudo_quantize_tensor_2bit(cur_w, bit=n_bit,q_group_size=group_size)
            else:
                q_w = pseudo_quantize_tensor(cur_w, bit=n_bit,q_group_size=group_size)
            #q_w = pseudo_quantize_tensor0(cur_w, bit=n_bit,group_size=group_size)
            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    #print(w_all.shape, best_max_val.shape)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)

    w_all = w_all.reshape(org_w_shape)
    return w_all





@torch.no_grad()
def auto_clip_lora2_test(
    W_O, w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    # w = w.to(torch.float64)
    # lora_W = lora_W.to(torch.float64)
    # input_feat = input_feat.to(torch.float64)

    #print(input_feat.shape)
    lw_ori= lora_W.T
    lora_W = lora_W.T
    lora_W = lora_W.contiguous()
    w = w.contiguous()
    input_feat = input_feat.contiguous()
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)
  
    lora_W = lora_W.reshape(lora_W.shape[0], 1, -1, group_size)
    oc_batch_size = 128 if w.shape[0] % 128 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w

        #print(w.shape)
        # if oz.abs().amax()>1e-4:
        #     print(oz.abs().amax())
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor_do(cur_w, bit=n_bit,q_group_size=group_size)

            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1

            #kl = calculate_kl_divergence_loss(org_w, q_w + lw, num_bins=16)
            

            err =  (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            # del cur_w
            # del q_w
            # del cur_out
            cur_best_idx = err < min_errs
            #print(cur_best_idx)
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        #del org_w
        best_max_val_all.append(best_max_val)
        # print(bset_i)
    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    # del input_feat
    # gc.collect()
    # torch.cuda.empty_cache()
    # del org_out
    # gc.collect()
    # torch.cuda.empty_cache()


    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)
    w_all = w_all.reshape(org_w_shape)

    import torch.nn.functional as F
    #print(n_bit, group_size)
    q_w_all = pseudo_quantize_tensor_do(w_all, bit=n_bit,q_group_size=group_size)
    lqw = q_w_all+lw_ori


    kl_loss = calculate_kl_divergence_loss(W_O, lqw, num_bins=200)
    #print(f"KL Divergence Loss: {kl_loss:.6f}")

    return w_all,kl_loss






@torch.no_grad()
def auto_clip_lora2_test_reduce(
    W_O, w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    # w = w.to(torch.float64)
    # lora_W = lora_W.to(torch.float64)
    # input_feat = input_feat.to(torch.float64)

    #print(input_feat.shape)
    lw_ori= lora_W.T
    lora_W = lora_W.T
    lora_W = lora_W.contiguous()
    w = w.contiguous()
    input_feat = input_feat.contiguous()
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)
  
    lora_W = lora_W.reshape(lora_W.shape[0], 1, -1, group_size)
    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w

        # if oz.abs().amax()>1e-4:
        #     print(oz.abs().amax())
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        min_reduce = torch.ones_like(org_max_val) * -100.0

        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group

        q_w0 = pseudo_quantize_tensor_do(w, bit=n_bit,q_group_size=group_size)
        q_out0 = (input_feat * (q_w0 + lw)).sum(dim=-1)
        err0 =  (q_out0 - org_out).pow(2).mean(dim=1).view(min_errs.shape)
        #kl0 = calculate_kl_divergence_loss(org_w, q_w0 + lw, num_bins=16)

        kl0 = calculate_kl_divergence_loss_batch(org_w, q_w0 + lw, num_bins=16)
        #print(w.shape, kl0_b.shape)
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor_do(cur_w, bit=n_bit,q_group_size=group_size)

            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1

            kl = calculate_kl_divergence_loss_batch(org_w, q_w + lw, num_bins=16)
            err =  (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            
            reduce = (0.3*(kl0-kl)/kl0) + (0.7*(err0-err)/err0)

            cur_best_idx = reduce > min_reduce
            #print(cur_best_idx)
            min_reduce[cur_best_idx] = reduce[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        #del org_w
        best_max_val_all.append(best_max_val)
        # print(bset_i)
    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    # del input_feat
    # gc.collect()
    # torch.cuda.empty_cache()
    # del org_out
    # gc.collect()
    # torch.cuda.empty_cache()


    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)
    w_all = w_all.reshape(org_w_shape)

    import torch.nn.functional as F
    #print(n_bit, group_size)
    q_w_all = pseudo_quantize_tensor_do(w_all, bit=n_bit,q_group_size=group_size)
    lqw = q_w_all+lw_ori


    kl_loss = calculate_kl_divergence_loss(W_O, lqw, num_bins=200)
    #print(f"KL Divergence Loss: {kl_loss:.6f}")

    return w_all,kl_loss



@torch.no_grad()
def auto_clip_lora2_test_reduce_norm(
    W_O, w, lora_W ,input_feat, norm, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    # w = w.to(torch.float64)
    # lora_W = lora_W.to(torch.float64)
    # input_feat = input_feat.to(torch.float64)

    #print(input_feat.shape)
    lw_ori= lora_W.T
    lora_W = lora_W.T
    lora_W = lora_W.contiguous()
    w = w.contiguous()
    input_feat = input_feat.contiguous()
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)
  
    lora_W = lora_W.reshape(lora_W.shape[0], 1, -1, group_size)
    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = (lw + w) * norm

        # if oz.abs().amax()>1e-4:
        #     print(oz.abs().amax())
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        min_reduce = torch.ones_like(org_max_val) * -100.0

        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group

        q_w0 = pseudo_quantize_tensor_do(w, bit=n_bit,q_group_size=group_size)
        q_out0 = (input_feat * (q_w0 + lw)).sum(dim=-1)
        err0 =  (q_out0 - org_out).pow(2).mean(dim=1).view(min_errs.shape)
        #kl0 = calculate_kl_divergence_loss(org_w, q_w0 + lw, num_bins=16)

        kl0 = calculate_kl_divergence_loss_batch(org_w, q_w0 + lw, num_bins=16)
        #print(w.shape, kl0_b.shape)
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor_do(cur_w, bit=n_bit,q_group_size=group_size)

            cur_out = (input_feat * ((q_w + lw) *norm)).sum(dim=-1)
            # co, 1, n_group, 1

            kl = calculate_kl_divergence_loss_batch(org_w, q_w + lw, num_bins=16)
            err =  (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            
            reduce = (0.3*(kl0-kl)/kl0) + (0.7*(err0-err)/err0)

            cur_best_idx = reduce > min_reduce
            #print(cur_best_idx)
            min_reduce[cur_best_idx] = reduce[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        #del org_w
        best_max_val_all.append(best_max_val)
        # print(bset_i)
    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    # del input_feat
    # gc.collect()
    # torch.cuda.empty_cache()
    # del org_out
    # gc.collect()
    # torch.cuda.empty_cache()


    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)
    w_all = w_all.reshape(org_w_shape)

    import torch.nn.functional as F
    #print(n_bit, group_size)
    q_w_all = pseudo_quantize_tensor_do(w_all, bit=n_bit,q_group_size=group_size)
    lqw = q_w_all+lw_ori


    kl_loss = calculate_kl_divergence_loss(W_O, lqw, num_bins=200)
    #print(f"KL Divergence Loss: {kl_loss:.6f}")

    return w_all,kl_loss




