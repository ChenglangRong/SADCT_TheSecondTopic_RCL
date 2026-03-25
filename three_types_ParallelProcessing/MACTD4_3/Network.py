import torch
import torch.nn as nn
import torch.nn.functional as F

# CTD4 Actor网络（保持确定性策略，与TD3一致，调整隐藏层维度）
class CTD4Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CTD4Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.temperature = 1.0

    def forward(self, x):
        x = torch.clamp(x, min=-200.0, max=200.0)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        probs = F.softmax(logits / self.temperature + 1e-8, dim=1)
        return probs

# CTD4 Critic网络（输出μ和σ，支持卡尔曼融合）
class CTD4Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CTD4Critic, self).__init__()
        # 合并状态和动作的特征提取
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        # 输出头：均值μ（无激活）和标准差σ（softplus确保为正）
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x)) + 1e-8  # 避免σ为0
        return mu, sigma