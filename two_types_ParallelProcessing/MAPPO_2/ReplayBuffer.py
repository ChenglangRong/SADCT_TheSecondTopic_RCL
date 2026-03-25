import random
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, old_log_prob, reward, next_state, done, mask, hidden):
        """存储单个完整样本，确保仅8个元素"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # 核心：直接存储8个元素的元组，不拆分内部结构
        self.buffer[self.position] = (state, action, old_log_prob, reward, next_state, done, mask, hidden)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """采样批量样本，返回的每个元素仍是完整的8字段样本"""
        batch = random.sample(self.buffer, batch_size)
        return batch  # 直接返回样本列表，不在此解包

    def __len__(self):
        return len(self.buffer)

    # 添加size方法，返回当前缓冲区中的样本数量
    def size(self):
        return len(self.buffer)