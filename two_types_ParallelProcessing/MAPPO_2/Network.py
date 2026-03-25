import torch
import torch.nn as nn
import torch.nn.functional as F

# PPO策略网络
class PolicyNetwork(nn.Module):
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
        logits = self.fc2(out)
        return logits

# PPO价值网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出状态价值
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        gru_out, hidden = self.gru(x, hidden)
        out = gru_out[:, -1, :]
        out = F.relu(self.fc1(out))
        value = self.fc2(out)
        return value