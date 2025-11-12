import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import logging
from hadamard_utils import *
from qlayer_name_utils import *
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from safetensors.torch import save_file
def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


class GPTQ_lora:

    def __init__(self, layer, lora):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.lora = lora
        self.perm = torch.zeros((self.columns), device=self.dev)
        #self.scale_index = torch.zeros((self.columns), device=self.dev, dtype= int)
    def add_batch(self, inp, out):

        # mean_feat = inp.abs().view(-1, inp.shape[-1]).mean(0)
        # mean_feat = mean_feat.pow(2.4)
        # mean_feat = mean_feat.clamp(min=1e-4)
        # scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

        # _, self.scale_index = torch.topk(scales, scales.shape[0])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())


    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        lora = self.lora.clone()
        W = W.float()
        lora = lora.float().to(W.device)

        res = W - lora
        

        abs_tensor = res.abs()  # 形状：[1000, 512]
        mean_abs_per_col = abs_tensor.mean(dim=0)  # 形状：[512]
        top_128_values, top_128_indices = torch.topk(-mean_abs_per_col, 128)
        print("\n最大的 128 个列（索引: 值）:")
        for idx, val in zip(top_128_indices, top_128_values):
            print(f"列 {idx.item()}: {-val.item():.4f}")

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(res)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        res[:, dead] = 0


        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = res[:, i1:i2].clone()

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(res[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            res[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        # # RE SVD
        # RR = res_c.float() - Q.float() + lora.float()
        # U, S, Vh = torch.linalg.svd(RR.float(), full_matrices=False)
        # truncated_rank = 40
        # U_trunc = U[:, :truncated_rank]
        # S_trunc = S[:truncated_rank]
        # Vh_trunc = Vh[:truncated_rank, :]
        # lora = (U_trunc @ torch.diag(S_trunc) @ Vh_trunc).to(self.layer.weight.data.dtype)
        # #  END

        QR = Q + lora.to(self.layer.weight.data.dtype)
        #print(QR)
        #self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = QR
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def fasterquant_rot(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        lora = self.lora.clone()
        W = W.float()
        lora = lora.float().to(W.device)

        res = W - lora
        
        H = self.H
        # Begin: construct partial permute rotate matrix.

        #perm = torch.sort(torch.diag(H), descending=True)
        
        partial_size = 256
        rotate_size = 128
        rot = block_diagonal_walsh_matrix(W.shape[1], rotate_size, W.device)
        abs_tensor = res.abs()
        mean_abs_per_col = abs_tensor.mean(dim=0) 
        top_128_values, top_128_indices = torch.topk(-mean_abs_per_col, W.shape[1])

        Ppermute = construct_partial_permutation_matrix_upper(top_128_indices, m = W.shape[1], dtype = W.dtype).to(W.device)
        diagI_rot = create_diagI_matrix_upper(rot, rot.shape[0]-partial_size).to(W.device).to(W.dtype)
        PD_rot = Ppermute @ diagI_rot
        # flag = torch.allclose(PD_rot.t() @ PD_rot, torch.eye(rot.shape[0], device=PD_rot.device, dtype=PD_rot.dtype), atol=1e-6)
        # print(flag)
        res = res @ PD_rot
        H = PD_rot.T @ H @ PD_rot

        # # END
        
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(res)

        
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        res[:, dead] = 0


        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = res[:, i1:i2].clone()

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(res[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            res[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        Q = (Q.to(PD_rot.dtype) @ PD_rot.T).to(self.layer.weight.data.dtype)
        QR = Q + lora.to(self.layer.weight.data.dtype)
        #print(QR)
        #self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = QR
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')


    def fasterquant_rot_new_perm(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        lora = self.lora.clone()
        W = W.float()
        lora = lora.float().to(W.device)

        res = W - lora
        
        H = self.H

        
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(res)

        
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        res[:, dead] = 0

        # Begin: construct partial permute rotate matrix.
        partial_size = 256
        rotate_size = 256
        rot = block_diagonal_walsh_matrix(W.shape[1], rotate_size, W.device)
        abs_tensor = res.abs()
        mean_abs_per_col = abs_tensor.mean(dim=0) 
        #top_values, top_indices = torch.topk(-mean_abs_per_col, W.shape[1])

        perm = torch.log2(torch.diag(H))
        perm = mean_abs_per_col/(perm - torch.min(perm) + 1)
        top_values, top_indices = torch.topk(-perm, W.shape[1])

        
        Ppermute = construct_partial_permutation_matrix_upper(top_indices, m = W.shape[1], dtype = W.dtype).to(W.device)
        diagI_rot = create_diagI_matrix_upper(rot, rot.shape[0]-partial_size).to(W.device).to(W.dtype)
        PD_rot = Ppermute @ diagI_rot
        # flag = torch.allclose(PD_rot.t() @ PD_rot, torch.eye(rot.shape[0], device=PD_rot.device, dtype=PD_rot.dtype), atol=1e-6)
        # print(flag)
        res = res @ PD_rot
        H = PD_rot.T @ H @ PD_rot
        # End

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = res[:, i1:i2].clone()

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(res[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            res[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        Q = (Q.to(PD_rot.dtype) @ PD_rot.T)
        QR = Q + lora
        

        # r2 = W - QR
        # U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        # truncated_rank = 32 
        # U_trunc = U[:, :truncated_rank]
        # S_trunc = S[:truncated_rank]
        # Vh_trunc = Vh[:truncated_rank, :]
        # lr2 = (U_trunc @ torch.diag(S_trunc) @ Vh_trunc)
        # QR = QR + lr2
        # print(S_trunc)
        #print(QR)
        #self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = QR.to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def fasterquant_rot_new_perm_split(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, ha_bsize=256, id_bsize = 256
    ):
        W = self.layer.weight.data.clone()
        if torch.any(torch.isnan(self.layer.weight)):
            logging.warning('NaN in ow')
            return
            raise ValueError('NaN in ow')   
 
        #print(self.lora["U"].dtype)
        lora = (self.lora["U"].to(W.dtype) @ torch.diag(self.lora["Si"])@ self.lora["V"].to(W.dtype))
        lora = (torch.diag(self.lora["Sa"]) @ lora).T
        #print(lora.shape)
        if torch.any(torch.isnan(lora)):
            logging.warning('NaN in lora')
            raise ValueError('NaN in lora')            
        W = W.float()
        lora = lora.float().to(W.device)

        res = W - lora
        
        H = self.H

        if torch.any(torch.isnan(res)):
            logging.warning('NaN in res')
            raise ValueError('NaN in res')        
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(res)

        
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        res[:, dead] = 0

        # Begin: construct partial permute rotate matrix.
        partial_size = id_bsize
        rotate_size = ha_bsize
        rot = block_diagonal_walsh_matrix(W.shape[1], rotate_size, W.device)
        abs_tensor = res.abs()
        mean_abs_per_col = abs_tensor.mean(dim=0) 
        #top_values, top_indices = torch.topk(-mean_abs_per_col, W.shape[1])

        perm = torch.log2(torch.diag(H))
        perm = mean_abs_per_col/(perm - torch.min(perm) + 1)
        top_values, top_indices = torch.topk(-perm, W.shape[1])

        
        Ppermute = construct_partial_permutation_matrix_upper(top_indices, m = W.shape[1], dtype = W.dtype).to(W.device)
        diagI_rot = create_diagI_matrix_upper(rot, rot.shape[0]-partial_size).to(W.device).to(W.dtype)
        PD_rot = Ppermute @ diagI_rot
        # flag = torch.allclose(PD_rot.t() @ PD_rot, torch.eye(rot.shape[0], device=PD_rot.device, dtype=PD_rot.dtype), atol=1e-6)
        # print(flag)


        # ablation+_+--------------------------
        # PD_rot = torch.eye(PD_rot.size(0), dtype=PD_rot.dtype, device=PD_rot.device)
        # PD_rot = random_hadamard_matrix(PD_rot.shape[0], device=PD_rot.device).to(W.dtype)
        # ablation+_+--------------------------
        res = res @ PD_rot
        H = PD_rot.T @ H @ PD_rot
        # End

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = res[:, i1:i2].clone()

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(res[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            res[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        
        Q = Q.reshape(self.layer.weight.shape)

        Q = (Q @ PD_rot.T)
        QR = Q.to(self.layer.weight.data.dtype) + lora.to(self.layer.weight.data.dtype)
        

        # r2 = W - QR
        # U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        # truncated_rank = 32 
        # U_trunc = U[:, :truncated_rank]
        # S_trunc = S[:truncated_rank]
        # Vh_trunc = Vh[:truncated_rank, :]
        # lr2 = (U_trunc @ torch.diag(S_trunc) @ Vh_trunc)
        # QR = QR + lr2
        # print(S_trunc)
        #print(QR)
        #self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = QR

        W = W.detach().cpu()
        lora = lora.detach().cpu()
        H = H.detach().cpu()
        Q = Q.detach().cpu()
        QR = QR.detach().cpu()
        PD_rot = PD_rot.detach().cpu()
        Ppermute = Ppermute.detach().cpu()
        diagI_rot = diagI_rot.detach().cpu()

        del W,lora, H, Q ,QR, PD_rot, Ppermute, diagI_rot

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保所有操作完成
            torch.cuda.empty_cache()  # 释放缓存显存
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            raise ValueError('NaN in weights')



    def fasterquant_rot_new_perm_split_save_p(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, ha_bsize=256, id_bsize = 256
    ):
        W = self.layer.weight.data.clone()
        lora = (self.lora["U"].to(W.dtype) @ torch.diag(self.lora["Si"])@ self.lora["V"].to(W.dtype))
        lora = (torch.diag(self.lora["Sa"]) @ lora).T
        #print(lora.shape)
        
        W = W.float()
        lora = lora.float().to(W.device)

        res = W - lora
        
        H = self.H

        
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(res)

        
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        res[:, dead] = 0

        # Begin: construct partial permute rotate matrix.
        partial_size = id_bsize
        rotate_size = ha_bsize
        rot = block_diagonal_walsh_matrix(W.shape[1], rotate_size, W.device)
        abs_tensor = res.abs()
        mean_abs_per_col = abs_tensor.mean(dim=0) 
        #top_values, top_indices = torch.topk(-mean_abs_per_col, W.shape[1])

        perm = torch.log2(torch.diag(H))
        perm = mean_abs_per_col/(perm - torch.min(perm) + 1)
        top_values, top_indices = torch.topk(-perm, W.shape[1])

        
        Ppermute = construct_partial_permutation_matrix_upper(top_indices, m = W.shape[1], dtype = W.dtype).to(W.device)
        diagI_rot = create_diagI_matrix_upper(rot, rot.shape[0]-partial_size).to(W.device).to(W.dtype)
        PD_rot = Ppermute @ diagI_rot
        # flag = torch.allclose(PD_rot.t() @ PD_rot, torch.eye(rot.shape[0], device=PD_rot.device, dtype=PD_rot.dtype), atol=1e-6)
        # print(flag)


        # ablation+_+--------------------------
        # PD_rot = torch.eye(PD_rot.size(0), dtype=PD_rot.dtype, device=PD_rot.device)
        # PD_rot = random_hadamard_matrix(PD_rot.shape[0], device=PD_rot.device).to(W.dtype)
        # ablation+_+--------------------------
        res = res @ PD_rot
        H = PD_rot.T @ H @ PD_rot
        # End

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = res[:, i1:i2].clone()

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(res[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            res[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        
        Q = Q.reshape(self.layer.weight.shape)

        S = np.array(top_indices.cpu())
        all_indices = set(range(W.shape[1]))
        S_set = set(S)
        front = list(S)
        back = sorted(all_indices - S_set)
        
        sigma = front+back
        self.perm = torch.tensor(sigma, device=self.dev)
        #Q = (Q @ PD_rot.T)
        self.pdrot = PD_rot
        QR = Q.to(self.layer.weight.data.dtype) + (lora @ PD_rot).to(self.layer.weight.data.dtype)
        


        self.pweight = QR.clone()#@ PD_rot.T.to(self.layer.weight.data.dtype)

        QR = QR @ PD_rot.T.to(self.layer.weight.data.dtype)



        #print(self.pweight.shape,QR.shape,self.pweight.dtype)
        P = torch.zeros((Ppermute.shape[-1], Ppermute.shape[-1]), dtype=Ppermute.dtype).to(Ppermute.device)
        for j, col_idx in enumerate(self.perm):
            P[col_idx, j] = 1.0  # 第 j 列是 e_{col_idx}

        bhw = block_diagonal_walsh_matrix(perm.shape[0], 256, device=Ppermute.device).half()
        nrot = create_diagI_matrix_upper(bhw, perm.shape[0] - 256 ).to(Ppermute.device).to(bhw.dtype)
        #print(torch.allclose(PD_rot, (P.to(bhw.dtype) @ nrot).to(PD_rot.dtype), atol=1e-3), QR.shape)



        # r2 = W - QR
        # U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        # truncated_rank = 32 
        # U_trunc = U[:, :truncated_rank]
        # S_trunc = S[:truncated_rank]
        # Vh_trunc = Vh[:truncated_rank, :]
        # lr2 = (U_trunc @ torch.diag(S_trunc) @ Vh_trunc)
        # QR = QR + lr2
        # print(S_trunc)
        #print(QR)
        #self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = QR
        #print(id_bsize, ha_bsize)
        bhw = bhw.detach().cpu()
        nrot = nrot.detach().cpu()
        W = W.detach().cpu()
        P = P.detach().cpu()
        lora = lora.detach().cpu()
        H = H.detach().cpu()
        Q = Q.detach().cpu()
        QR = QR.detach().cpu()
        PD_rot = PD_rot.detach().cpu()
        Ppermute = Ppermute.detach().cpu()
        diagI_rot = diagI_rot.detach().cpu()

        del W,lora, H, Q ,QR, PD_rot, Ppermute, diagI_rot, bhw, nrot, P

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保所有操作完成
            torch.cuda.empty_cache()  # 释放缓存显存
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd_lora(model, loras, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # model = model.to(dev)

    # for module in model.modules():
    #     module.to(dev)
    layer_kwargs = {}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            layer_kwargs.update(kwargs)
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    #print(dataloader)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    #layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    # sequential = [
    #             ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
    #             ['self_attn.o_proj.module'],
    #             ['mlp.up_proj.module', 'mlp.gate_proj.module'],
    #             ['mlp.down_proj.module']
    #         ]

    # sequential = [
    #             ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
    #             ['self_attn.o_proj'],
    #             ['mlp.up_proj', 'mlp.gate_proj'],
    #             ['mlp.down_proj']
    #         ]
    if isinstance(layers[0], OPTDecoderLayer):
        sequential = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"]
    elif isinstance(layers[0], LlamaDecoderLayer):
        sequential = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]
    elif isinstance(layers[0], MixtralDecoderLayer):
        sequential = MixtralQuantLayerA
    elif isinstance(layers[0], Qwen2DecoderLayer):
        sequential = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]
    print(sequential)
    for i in range(len(layers)): 

        lora_layer = loras[i]
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue

                gptq[name] = GPTQ_lora(subset[name], lora_layer[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant_rot_new_perm_split(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False, id_bsize = args.id_bsize, ha_bsize = args.ha_bsize
                )

                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers


@torch.no_grad()
def gptq_fwrd_lora_savep(model, loras, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # model = model.to(dev)

    # for module in model.modules():
    #     module.to(dev)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    #print(dataloader)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    #layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    # sequential = [
    #             ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
    #             ['self_attn.o_proj.module'],
    #             ['mlp.up_proj.module', 'mlp.gate_proj.module'],
    #             ['mlp.down_proj.module']
    #         ]

    # sequential = [
    #             ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
    #             ['self_attn.o_proj'],
    #             ['mlp.up_proj', 'mlp.gate_proj'],
    #             ['mlp.down_proj']
    #         ]
    if isinstance(layers[0], OPTDecoderLayer):
        sequential = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"]
    elif isinstance(layers[0], LlamaDecoderLayer):
        sequential = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]
    elif isinstance(layers[0], MixtralDecoderLayer):
        sequential = MixtralQuantLayerA
    
    perm_safetensor = {}
    
    for i in range(len(layers)): 

        # if i>4:
        #     break
        lora_layer = loras[i]
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}
            pdrot_safetensor = {}
            pweight = {}
            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue

                gptq[name] = GPTQ_lora(subset[name], lora_layer[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant_rot_new_perm_split_save_p(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False, id_bsize = args.id_bsize, ha_bsize = args.ha_bsize
                )

                full_name = 'model.layers.%d.%s' % (i, name)
                if full_name == "model.layers.0.self_attn.k_proj":
                    print(gptq[name].pdrot.cpu())
                if full_name == "model.layers.1.mlp.up_proj":
                    print(gptq[name].pdrot.cpu())
                perm_safetensor[full_name] = gptq[name].perm.cpu()
                pdrot_safetensor[full_name] = gptq[name].pdrot.cpu()
                pweight[full_name] = gptq[name].pweight.cpu()
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()
            save_file(pdrot_safetensor, "/rjs/ghyx/data/perm/"+str(i)+".safetensors")
            save_file(pweight, "/rjs/ghyx/data/perm/pweight"+str(i)+".safetensors")
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        


        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    
    #print(pdrot_safetensor)
    save_file(perm_safetensor, "/rjs/ghyx/data/perm/perm.safetensors")
    model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers



@torch.no_grad()
def gptq_fwrd_lora_savep2(model, loras, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # model = model.to(dev)

    # for module in model.modules():
    #     module.to(dev)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    #print(dataloader)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    #layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    # sequential = [
    #             ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
    #             ['self_attn.o_proj.module'],
    #             ['mlp.up_proj.module', 'mlp.gate_proj.module'],
    #             ['mlp.down_proj.module']
    #         ]

    # sequential = [
    #             ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
    #             ['self_attn.o_proj'],
    #             ['mlp.up_proj', 'mlp.gate_proj'],
    #             ['mlp.down_proj']
    #         ]
    if isinstance(layers[0], OPTDecoderLayer):
        sequential = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"]
    elif isinstance(layers[0], LlamaDecoderLayer):
        sequential = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]
    elif isinstance(layers[0], MixtralDecoderLayer):
        sequential = MixtralQuantLayerA
    
    perm_safetensor = {}
    
    for i in range(len(layers)): 

        # if i>4:
        #     break
        lora_layer = loras[i]
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        pweight = {}
        for names in sequential:
            subset = {n: full[n] for n in names}
            
            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue

                gptq[name] = GPTQ_lora(subset[name], lora_layer[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant_rot_new_perm_split_save_p(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False, id_bsize = args.id_bsize, ha_bsize = args.ha_bsize
                )

                full_name = 'model.layers.%d.%s' % (i, name)
                if full_name == "model.layers.0.self_attn.k_proj":
                    print(gptq[name].pdrot.cpu())
                if full_name == "model.layers.1.mlp.up_proj":
                    print(gptq[name].pdrot.cpu())
                perm_safetensor[full_name] = gptq[name].perm.cpu()
                pweight[name] = gptq[name].pweight.cpu()
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                subset[name].weight.data = pweight[name].to(subset[name].weight.data.dtype).to(subset[name].weight.device)

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    save_file(perm_safetensor, "/rjs/ghyx/data/perm/perm.safetensors")
    model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers


       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        if i>3:
            break
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    cleanup_memory(verbos=True)
    return quantizers
