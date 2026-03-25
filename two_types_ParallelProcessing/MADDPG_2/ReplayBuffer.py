import random
import numpy as np
import collections
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from pylab import mpl


# -------------------------------------------------
#   经验回放池
# -------------------------------------------------
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    # 修改 add 方法，增加 mask 参数
    def add(self, state, action, reward, next_state, done, mask):  # 现在接受6个参数（加self共7个）
        self.buffer.append((state, action, reward, next_state, done, mask))

    # 修改 sample 方法，返回 mask
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask = zip(*transitions)  # 增加 mask
        return np.array(state), action, reward, np.array(next_state), done, mask  # 返回 mask

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)