import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 设置X轴刻度间隔为16
# 定义百分比格式化函数
def to_percent(y, position):
    # 在此处可以乘以100或其他操作来确保数据已经是百分比形式
    return f'{y * 100:.2f}%'

def to_percent_2(y, position):
    return f'{y:.2f}'

def draw_mse(x,max_abs,mse,srank0,fig_name):

    delta_mse = [mse[i] - mse[i + 1] for i in range(len(mse) - 1)]
    delta_mse.append(delta_mse[-1])  # Append the last value to maintain the same length

    mse = [value / mse[0] for value in mse]
    max_abs = [value / max_abs[0] for value in max_abs]


    fig, ax1 = plt.subplots(figsize=(6, 5))

    # 绘制第一组数据
    color = 'black'
    ax1.set_xlabel('The $Rank$ of low-rank quantization', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Relative Error $\mathbb{E}$', color=color, fontweight='bold', fontsize=12)
    ax1.plot(x, mse, color=color)
    ax1.tick_params(axis='y', labelcolor=color, size=8)



    # 使用FuncFormatter应用百分比格式化函数到第一个Y轴
    formatter = FuncFormatter(to_percent)
    ax1.yaxis.set_major_formatter(formatter)


    ax1.set_xticks(range(0, 129, 16))

    # 创建第二个坐标轴
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Maximum absolute value', color=color, fontweight='bold', fontsize=12)  
    ax2.plot(x, max_abs, color=color)
    ax2.tick_params(axis='y', labelcolor=color, size=8)
    # # 使用FuncFormatter应用百分比格式化函数到第一个Y轴
    formatter = FuncFormatter(to_percent_2)
    ax2.yaxis.set_major_formatter(formatter)

    plt.axvline(x=srank0, color='blue', linestyle='--', linewidth=1, label='optimal rank')
    # 设置两个Y轴的刻度数量均为8
    for ax in [ax1, ax2]:
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.linspace(start, end, 8))


    for ax in [ax1, ax2]:
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.linspace(start, end, 8))
        # 如果需要，可以对第二个Y轴也启用网格线（重复上述grid()调用）
        ax.grid(True, which='major', linestyle=':', color='gray')
        ax.grid(True, which='minor', linestyle=':', color='gray')


    fig.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.show()



    # 调整布局
    plt.tight_layout()
    # # 保存为PDF
    # plt.savefig('mse_abs.pdf', format='pdf')
    # 保存为jpg
    plt.savefig(fig_name+'.png', dpi=300, bbox_inches='tight')
    # # 保存图表为SVG格式
    # plt.savefig('mse_abs.svg', format='svg')
    plt.close()