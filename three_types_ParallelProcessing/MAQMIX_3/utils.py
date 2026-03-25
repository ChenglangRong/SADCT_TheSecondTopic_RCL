import os
import random
import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt
import torch


def moving_average(a, window_size):
    if window_size <= 1:
        return np.array(a)

    # 计算中间部分的滑动平均
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

    # 计算前半部分的滑动平均（处理边界）
    begin_len = (window_size - 1) // 2
    begin = []
    for i in range(1, begin_len + 1):
        begin.append(np.mean(a[:2 * i]))
    begin = np.array(begin)

    # 计算后半部分的滑动平均（处理边界）
    end_len = (window_size - 1) - begin_len
    end = []
    for i in range(1, end_len + 1):
        end.append(np.mean(a[-2 * i:]))
    end = np.array(end)[::-1]  # 反转以保持顺序

    return np.concatenate((begin, middle, end))

def scatter_method(x_list,  y_list, title, ylabel, figure_file):
    plt.figure()
    plt.scatter(x_list,  y_list, color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file, dpi=300)  # dpi=300调节分辨率，默认为100
    plt.show()

def plot_method(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file, dpi=300)  # dpi=300调节分辨率，默认为100
    plt.show()

def save_data(data,data_file_path):
    df = pd.DataFrame(data)
    # 导出到 Excel 文件
    df.to_excel(data_file_path, index=False)

def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)