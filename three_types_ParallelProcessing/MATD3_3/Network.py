import torch
import torch.nn as nn
import torch.nn.functional as F

# DDPG Actor网络（离散动作）
class TD3Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(TD3Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # 层归一化，稳定训练
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # 层归一化
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.temperature = 1.0  # 可动态调整的温度系数

    def forward(self, x):
        # 输入裁剪：限制网络输入的范围（与状态处理的范围一致）
        x = torch.clamp(x, min=-200.0, max=200.0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        # 移除logits的过度裁剪（原代码的0.1~0.9可能导致分布异常）
        # 改为宽松裁剪，避免softmax计算溢出
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        # 温度系数调整（可选，避免分布过于尖锐）
        probs = F.softmax(logits / self.temperature + 1e-8, dim=1)
        return probs

# DDPG Critic网络
class TD3Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(TD3Critic, self).__init__()
        # 状态输入分支
        self.fc1 = nn.Linear(state_dim, hidden_dim//2)
        # 动作输入分支
        self.fc2 = nn.Linear(action_dim, hidden_dim//2)
        # 合并层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(action))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc3(x))
        return self.fc4(x)