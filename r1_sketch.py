import torch
import torch.nn as nn
import argparse
import os
import json
import math
#


from numpy import random



qtype = torch.float16
def find_group_max_abs_value(A,group_size):
    # 计算矩阵 A 的绝对值
    A = A.reshape(-1, group_size)
    abs_A = torch.abs(A)
    max_abs_value = abs_A.amax(dim=1, keepdim=True)
    sum_elements = torch.sum(max_abs_value.to(dtype=torch.float32))  
    result = sum_elements / max_abs_value.size(0)

    return result

def find_group_max_diff_value(A,group_size):
    # 计算矩阵 A 的绝对值
    if group_size > 0:
        A = A.reshape(-1, group_size)
    diff_value = A.amax(dim=1, keepdim=True) - A.amin(dim=1, keepdim=True)
    sum_elements = torch.sum(diff_value.to(dtype=torch.float32).pow(2))
    result = torch.sqrt(sum_elements / diff_value.size(0))
    return result

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
    x = x.to(torch.float64)
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


def compute_r1sketch_retS(A,iter = 1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = A.shape
    
    x_numpy = random.normal(loc=0, scale=1, size=(n))
    # 生成一个长度为 n 的随机正态分布向量 x
    
    x = torch.from_numpy(x_numpy)
    x = x.to(torch.float64)
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

    sVal = normQ/normP

    A_R = A_R*Var_AR
    A_L = A_L*Var_AL
    # SK = np.outer(A_L, A_R)
    return A_L, A_R, sVal


def compute_r1sketch_fp8(A,iter = 1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = A.shape
    
    x_numpy = random.normal(loc=0, scale=1, size=(n))
    # 生成一个长度为 n 的随机正态分布向量 x
    
    x = torch.from_numpy(x_numpy)
    x = x.to(torch.float64)
    x = x.to(device)
    y = torch.matmul(A, x)

    for i in range(iter):
        tmp = torch.matmul(A.T, y)
        y = torch.matmul(A, tmp)


    A_L = y
    A_R = torch.matmul(A.T, A_L)


    normP = torch.norm(A_L, p=2)
    normQ = torch.norm(A_R, p=2)

    S = normQ/normP
    Var_AL = 1.0/normP
    Var_AR = 1.0/normQ


    A_R = A_R*Var_AR
    A_L = A_L*Var_AL

    A_L = A_L.to(torch.float8_e4m3fn)
    A_L = A_L.to(torch.float16)

    A_R = A_R.to(torch.float8_e4m3fn)
    A_R = A_R.to(torch.float16)

    A_L = A_L*S
    # SK = np.outer(A_L, A_R)
    return A_L, A_R

def compute_r1sketch_fp8_ret(A,iter = 1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = A.shape
    
    x_numpy = random.normal(loc=0, scale=1, size=(n))
    # 生成一个长度为 n 的随机正态分布向量 x
    
    x = torch.from_numpy(x_numpy)
    x = x.to(torch.float64)
    x = x.to(device)
    y = torch.matmul(A, x)

    for i in range(iter):
        tmp = torch.matmul(A.T, y)
        y = torch.matmul(A, tmp)


    A_L = y
    A_R = torch.matmul(A.T, A_L)


    normP = torch.norm(A_L, p=2)
    normQ = torch.norm(A_R, p=2)

    S = normQ/normP
    Var_AL = 1.0/normP
    Var_AR = 1.0/normQ


    A_R = A_R*Var_AR
    A_L = A_L*Var_AL

    A_L = A_L.to(torch.float8_e5m2)
    A_R = A_R.to(torch.float8_e5m2)



    # SK = np.outer(A_L, A_R)
    return A_L, A_R, S

def get_best_sketch(weights, bits, ratio=0.01, max_sketch_iter = 4, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float64)
    if weights.dtype == torch.bfloat16:
        weights = weights.to(torch.float64)
    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)

    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = 8#int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
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
            #print(f"max_absW: {max_absW}, rank: {i+1},K: {K}, Q:{Q}, P:{P}")
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)

            # if(max_absW<1e-4):
            #     work_rank = i
            #     break                
            if(i>=max_ptr):
                if(((max_iter[i-max_ptr]-max_iter[i])/max_iter[i-max_ptr])<0.1):
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
        VS_L_16 = VS_L#.to(qtype)
        VS_R_16 = VS_R#.to(qtype)
        weight_cp = weight_cp - torch.matmul(VS_L_16.T,VS_R_16)
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank
    #return weight_cp,VS_L,VS_R,max_absW_0,max_iter[work_rank-1]

def get_best_sketch_fp8(weights, bits, ratio=0.01, max_sketch_iter = 4, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float64)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)

    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = 8#int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L,r1_R = compute_r1sketch_fp8(weights,max_sketch_iter)
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
            r1_L,r1_R = compute_r1sketch_fp8(weights,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            P = max_absW_0/max_absW
            K = 1.0+(8.0*(i+1)*(row+col)/(1.0*bits*row*col))
            Q = (bits + math.log(P,2) )/(1.0*bits)
            #print(f"max_absW: {max_absW}, rank: {i+1},K: {K}, Q:{Q}, P:{P}")
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)

            # if(max_absW<1e-4):
            #     work_rank = i
            #     break                
            if(i>=max_ptr):
                if(((max_iter[i-max_ptr]-max_iter[i])/max_iter[i-max_ptr])<0.1):
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

#返回值U,S,V， UV为fp8
def get_best_sketch_fp8_ret(weights, bits, ratio=0.01, max_sketch_iter = 8, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float64)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)

    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = 16#int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0

    S_arr = []
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L,r1_R,S = compute_r1sketch_fp8_ret(weights,max_sketch_iter)

            r1_L_FP16 = r1_L.to(torch.float16)
            r1_R_FP16 = r1_R.to(torch.float16)
            r1_L_FP16 = r1_L_FP16* S

            r1_matrix = torch.outer(r1_L_FP16, r1_R_FP16)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            S_arr.append(S.item())
        VS_L = skethc_L[:work_rank]
        VS_R = skethc_R[:work_rank]
        S_arr = S_arr[:work_rank]
        max_now = max_iter[work_rank-1]
    else:
        for i in range(0,min_rank):
            r1_L,r1_R,S = compute_r1sketch_fp8_ret(weights,max_sketch_iter)

            r1_L_FP16 = r1_L.to(torch.float16)
            r1_R_FP16 = r1_R.to(torch.float16)
            r1_L_FP16 = r1_L_FP16* S
            r1_matrix = torch.outer(r1_L_FP16, r1_R_FP16)


            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            P = max_absW_0/max_absW
            K = 1.0+(12.0*(i+1)*(row+col)/(1.0*bits*row*col))
            Q = (bits + math.log(P,2) )/(1.0*bits)
            #print(f"max_absW: {max_absW}, rank: {i+1},K: {K}, Q:{Q}, P:{P}")
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            S_arr.append(S.item())
            # if(max_absW<1e-4):
            #     work_rank = i
            #     break                
            if(i>=max_ptr):
                if(((max_iter[i-max_ptr]-max_iter[i])/max_iter[i-max_ptr])<0.1):
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
            VS_L = skethc_L[:work_rank]
            VS_R = skethc_R[:work_rank]
            S_arr = S_arr[:work_rank]
            max_now = max_iter[work_rank-1]

        if(work_rank!=0 and max_absW_0<=max_iter[work_rank-1]):
            work_rank = 0
            VS_L.zero_()
            VS_R.zero_()
            S_arr.zero_()
            max_now = max_absW_0

    #plot_weight_histogram(weights, "W2")
    if work_rank!=0:
        weight_cp = weight_cp
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L,VS_R,S_arr,max_absW_0,max_now,work_rank
    #return weight_cp,VS_L,VS_R,max_absW_0,max_iter[work_rank-1]


#返回值加了个奇异值
def get_best_sketch_retS(weights, bits, ratio=0.01, max_sketch_iter = 4, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float64)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)

    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = 8#int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0

    sVals = []
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L,r1_R,sVal = compute_r1sketch_retS(weights,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            sVals.append(sVal.cpu().item())
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
        max_now = max_iter[work_rank-1]
    else:
        for i in range(0,min_rank):
            r1_L,r1_R,sVal = compute_r1sketch(weights,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            P = max_absW_0/max_absW
            K = 1.0+(16.0*(i+1)*(row+col)/(1.0*bits*row*col))
            Q = (bits + math.log(P,2) )/(1.0*bits)
            #print(f"max_absW: {max_absW}, rank: {i+1},K: {K}, Q:{Q}, P:{P}")
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            sVals.append(sVal)
            # if(max_absW<1e-4):
            #     work_rank = i
            #     break                
            if(i>=max_ptr):
                if(((max_iter[i-max_ptr]-max_iter[i])/max_iter[i-max_ptr])<0.1):
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
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank,sVals



def get_group_diff_best_sketch(weights, bits, group_size = 128,ratio=0.01, max_sketch_iter = 2, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float64)

    
    max_absW_0 = find_group_max_diff_value(weights,group_size)

    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = 8#int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
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
            max_absW = find_group_max_diff_value(weights,group_size)
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
            max_absW = find_group_max_diff_value(weights,group_size)
            max_iter[i] = max_absW
            P = max_absW_0/max_absW
            K = 1.0+(16.0*(i+1)*(row+col)/(1.0*bits*row*col))
            Q = (bits + math.log(P,2) )/(1.0*bits)
            #print(f"max_absW: {max_absW}, rank: {i+1},K: {K}, Q:{Q}, P:{P}")
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)

            # if(max_absW<1e-4):
            #     work_rank = i
            #     break

            if(P>4e2):
                work_rank = i
                break                
            if(i>=max_ptr):
                if(((max_iter[i-max_ptr]-max_iter[i])/max_iter[i-max_ptr])<0.1):
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
        weights = weights.to(torch.float64)
    if weights.dtype == torch.bfloat16:
        weights = weights.to(torch.float64)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)
    U, s, Vt = torch.linalg.svd(weights, full_matrices=False)


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
        VS_L_16 = VS_L.to(torch.float64)#.to(torch.float16)
        VS_R_16 = VS_R.to(torch.float64)#.to(torch.float16)
        weight_cp = weight_cp - torch.matmul(VS_L_16.T,VS_R_16)
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank
    #return weight_cp,VS_L,VS_R,max_absW_0,max_iter[work_rank-1]


def get_mse_best_sketch(w_res_scale, w_q ,w_org, feat_scale, input_feat, bits, ratio=0.01, max_sketch_iter = 2, fix_rank = 0):
    row = w_res_scale.size(0)
    col = w_res_scale.size(1)
    min_rank = min(row,col)

    out_org = input_feat @ w_org.T

    weight_cp = w_res_scale
    if w_res_scale.dtype == torch.float16:
        w_res_scale = w_res_scale.to(torch.float64)

    
    max_absW_0 = find_max_abs_value(w_res_scale)

    skethc_L = []
    skethc_R = []

    best_rela_loss = 0.0
    best_error = float("inf")
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L,r1_R = compute_r1sketch(w_res_scale,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            w_res_scale = w_res_scale - r1_matrix
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
    else:
        for i in range(0,min_rank):
            r1_L,r1_R = compute_r1sketch(w_res_scale,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            w_res_scale = w_res_scale - r1_matrix

            w_res_scale_lora = torch.diag(feat_scale.float()).inverse().half() @ (weight_cp.half() - w_res_scale.half())
            lora_out = input_feat @ (w_q.T + w_res_scale_lora)
            loss = (out_org.to(input_feat.device) - lora_out.to(input_feat.device)).float().pow(2).sum().item()
            rela_loss = loss/ (out_org.to(input_feat.device).float().pow(2).sum().item())
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)

            K = 1.0+(16.0*(i+1)*(row+col)/(1.0*bits*row*col))
            #print(rela_loss)

            if loss < best_error:
                best_rela_loss = rela_loss
                work_rank = i
            if(K>(1.0+ratio)):
                work_rank = i
                break
        print(best_rela_loss, work_rank)

        
        if(work_rank == 0):
            work_rank = 1

        if(work_rank>=1):
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])

    if work_rank!=0:
        VS_L_16 = VS_L.to(torch.float16)
        VS_R_16 = VS_R.to(torch.float16)
        weight_cp = weight_cp - torch.matmul(VS_L_16.T,VS_R_16)
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank
    #return weight_cp,VS_L,VS_R,max_absW_0,max_iter[work_rank-1]


