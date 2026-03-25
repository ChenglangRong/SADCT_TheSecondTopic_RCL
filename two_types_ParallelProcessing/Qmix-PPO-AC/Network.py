import torch
import torch.nn as nn
import torch.nn.functional as F

# 原有Q网络保持不变
class AgentQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        gru_out, hidden = self.gru(x, hidden)
        out = gru_out[:, -1, :]
        out = F.relu(self.fc1(out))
        q_values = self.fc2(out)
        return q_values, hidden

# 新增PPO策略网络
class PolicyNetwork(nn.Module):
    """策略网络（输出动作概率分布）"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        gru_out, hidden = self.gru(x, hidden)
        out = gru_out[:, -1, :]
        out = F.relu(self.fc1(out))
        logits = self.fc2(out)  # 不使用softmax，交给Categorical处理
        return logits

# 原有混合网络保持不变
class HyperNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim_mixing):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim_mixing)
        self.fc2 = nn.Linear(hidden_dim_mixing, n_agents * hidden_dim_mixing)
        self.fc3 = nn.Linear(hidden_dim_mixing, hidden_dim_mixing)
        self.fc4 = nn.Linear(hidden_dim_mixing, hidden_dim_mixing)
        self.n_agents = n_agents
        self.hidden_dim_mixing = hidden_dim_mixing

    def forward(self, s):
        x = F.relu(self.fc1(s))
        hidden_weight = self.fc2(x).view(-1, self.n_agents, self.hidden_dim_mixing)
        hidden_bias = self.fc3(x).view(-1, 1, self.hidden_dim_mixing)
        output_weight = self.fc4(x).view(-1, self.hidden_dim_mixing, 1)
        output_bias = torch.zeros_like(output_weight[..., 0])
        return hidden_weight, hidden_bias, output_weight, output_bias

class MixingNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim_mixing=32):
        super().__init__()
        self.n_agents = n_agents
        self.hypernet = HyperNetwork(state_dim, n_agents, hidden_dim_mixing)
        self.elu = nn.ELU()

    def forward(self, q_values, s):
        batch_size = q_values.size(0)
        q_values = q_values.view(batch_size, 1, self.n_agents)
        hidden_weight, hidden_bias, output_weight, output_bias = self.hypernet(s)
        hidden = self.elu(torch.bmm(q_values, hidden_weight) + hidden_bias)
        q_tot = torch.bmm(hidden, output_weight) + output_bias
        return q_tot.squeeze(1)