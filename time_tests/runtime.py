import torch
import time
import numpy as np

# -----------------------------
# 用户可配置参数
# -----------------------------
m = 4096
n = 16384
b = 2048
srank = 16
# 是否使用 GPU (cuda) 或 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 计时运行次数 (取平均值更准确)
num_runs = 100

# -----------------------------
# 生成矩阵 W 和 X
# -----------------------------
# 使用 PyTorch (推荐，支持 GPU)
W = torch.randn(m, n, device=device, dtype=torch.float16)
X = torch.randn(n, b, device=device, dtype=torch.float16)

# (可选) 如果你想用 NumPy (仅限 CPU)
# W = np.random.randn(m, n).astype(np.float32)
# X = np.random.randn(n, b).astype(np.float32)

print(f"矩阵维度: W({m}x{n}), X({n}x{b})")

# -----------------------------
# 执行矩阵乘法并计时
# -----------------------------
# 预热：让 GPU/CPU 稳定下来
for _ in range(10):
    Y = W @ X
torch.cuda.synchronize()  # 等待 GPU 操作完成 (如果使用 GPU)

# 正式计时
start_time = time.time()

for _ in range(num_runs):
    Y = W @ X
    # 如果使用 GPU，必须同步以确保计算完成
    if device == 'cuda':
        torch.cuda.synchronize()

end_time = time.time()

# 计算平均执行时间
total_time = end_time - start_time
avg_time = total_time / num_runs

# -----------------------------
# 计算 TFLOPS
# -----------------------------
# 总浮点运算次数 (FLOPs)
# 矩阵乘法: m * b * (2 * n - 1) ≈ 2 * m * n * b
flops = 2.0 * m * n * b

# TFLOPS = (FLOPs) / (时间 in seconds) / 1e12
tflops = flops / avg_time / 1e12

# -----------------------------
# 输出结果
# -----------------------------
print(f"平均执行时间: {avg_time*1000:.4f} ms")
print(f"性能: {tflops:.2f} TFLOPS")




# U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
# # 截取前 rank 个奇异值和对应的向量
# U_trunc = U[:, :srank].half()
# S_trunc = S[:srank].half()
# Vh_trunc = Vh[:srank, :].half()

# US = U_trunc @ torch.diag(S_trunc)

US = torch.randn(m, srank, device=device, dtype=torch.float16)
Vh_trunc = torch.randn(srank, n, device=device, dtype=torch.float16)


for _ in range(10):
    P = Vh_trunc @ X
    Y = US @ P
torch.cuda.synchronize()  


# 正式计时
start_time = time.time()
for _ in range(num_runs):
    P = Vh_trunc @ X
    # Y = US @ P

    torch.cuda.synchronize()
end_time = time.time()

# 计算平均执行时间
total_time = end_time - start_time
avg_time = total_time / num_runs
print(f"平均lora执行时间: {avg_time*1000:.4f} ms")

import fast_hadamard_transform

X = X.T
scale = 1#/(n**0.5)
for _ in range(10):
    fast_hadamard_transform.hadamard_transform(X, scale)
torch.cuda.synchronize()

# 正式计时
print(X.shape)
start_time = time.time()
for _ in range(num_runs):
    fast_hadamard_transform.hadamard_transform(X, scale)
    # 如果使用 GPU，必须同步以确保计算完成
    #if device == 'cuda':
    torch.cuda.synchronize()
end_time = time.time()

# 计算平均执行时间
total_time = end_time - start_time
avg_time = total_time / num_runs
print(f"平均HW执行时间: {avg_time*1000:.4f} ms")

# from flash_attn.utils.benchmark import benchmark_forward, pytorch_profiler
# batch_size = 16
# seqlen = 2048
# dim = 16384 * 2
# dtype = torch.float16
# device = "cuda"

# torch.random.manual_seed(0)
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(fast_hadamard_transform.hadamard_transform, x, desc="Hadamard transform")
# pytorch_profiler(fast_hadamard_transform.hadamard_transform, x)

def block_hadamard_transform(X, block_size, scale):

    n, b = X.shape
    assert b % block_size == 0, "b must be divisible by block_size"
    num_blocks = b // block_size
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        fast_hadamard_transform.hadamard_transform(X[:,start:end], scale)
    return X


bsize = 256

for _ in range(10):
    block_hadamard_transform(X, bsize, scale)
torch.cuda.synchronize()
start_time = time.time()
for _ in range(num_runs):
    block_hadamard_transform(X, bsize, scale)
    # 如果使用 GPU，必须同步以确保计算完成
    #if device == 'cuda':
    torch.cuda.synchronize()
end_time = time.time()

# 计算平均执行时间
total_time = end_time - start_time
avg_time = total_time / num_runs
print(f"平均bHW执行时间: {avg_time*1000:.4f} ms")
