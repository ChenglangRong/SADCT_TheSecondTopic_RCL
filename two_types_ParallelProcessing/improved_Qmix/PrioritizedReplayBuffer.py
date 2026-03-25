import random
import numpy as np
import collections
import torch
import torch.nn.functional as F


class PrioritizedReplayBuffer:
    """带优先级的经验回放池"""

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha  # 优先级权重

    def add(self, state, action, reward, next_state, done, mask):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, mask))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done, mask)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # 计算概率
        probs = prios ** self.alpha
        probs /= probs.sum()

        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 计算权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done, mask = zip(*samples)
        return (np.array(state), action, reward, np.array(next_state), done, mask,
                indices, torch.tensor(weights, dtype=torch.float))

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6  # 避免为0

    def size(self):
        return len(self.buffer)