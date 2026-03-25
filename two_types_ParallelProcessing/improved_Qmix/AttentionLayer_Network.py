import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AttentionLayer(nn.Module):
    """注意力层，用于动态计算智能体权重"""

    def __init__(self, n_agents, state_dim, hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.query = nn.Linear(state_dim, hidden_dim)
        self.key = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(n_agents, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_agents)

    def forward(self, q_values, state):
        # q_values: (batch_size, n_agents)
        # state: (batch_size, state_dim)

        # 计算注意力权重
        query = self.query(state).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        key = self.key(state).unsqueeze(2)  # (batch_size, hidden_dim, 1)
        energy = torch.bmm(query, key).squeeze(2)  # (batch_size, 1)

        # 与Q值交互
        value = self.value(q_values)  # (batch_size, hidden_dim)
        attention = torch.tanh(energy + value)  # (batch_size, hidden_dim)
        attention_weights = F.softmax(self.fc(attention), dim=1)  # (batch_size, n_agents)

        return attention_weights


class HyperNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim_mixing=32):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim_mixing = hidden_dim_mixing

        # 加入注意力层
        self.attention = AttentionLayer(n_agents, state_dim, hidden_dim_mixing)

        self.fc_hidden_weight = nn.Linear(state_dim, n_agents * hidden_dim_mixing)
        self.fc_hidden_bias = nn.Linear(state_dim, hidden_dim_mixing)
        self.fc_output_weight = nn.Linear(state_dim, hidden_dim_mixing * 1)
        self.fc_output_bias = nn.Sequential(
            nn.Linear(state_dim, hidden_dim_mixing),
            nn.ReLU(),
            nn.Linear(hidden_dim_mixing, 1)
        )

    def forward(self, s, q_values):
        batch_size = s.size(0)

        # 获取注意力权重
        attention_weights = self.attention(q_values, s)  # (batch_size, n_agents)

        # 隐藏层权重：结合注意力权重
        hidden_weight = self.fc_hidden_weight(s).view(batch_size, self.n_agents, self.hidden_dim_mixing)
        hidden_weight = torch.abs(hidden_weight) * attention_weights.unsqueeze(2)  # 应用注意力

        hidden_bias = self.fc_hidden_bias(s).view(batch_size, 1, self.hidden_dim_mixing)
        output_weight = self.fc_output_weight(s).view(batch_size, self.hidden_dim_mixing, 1)
        output_weight = torch.abs(output_weight)
        output_bias = self.fc_output_bias(s).view(batch_size, 1, 1)

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

        # 传入q_values用于注意力计算
        hidden_weight, hidden_bias, output_weight, output_bias = self.hypernet(s, q_values.squeeze(1))

        hidden = self.elu(torch.bmm(q_values, hidden_weight) + hidden_bias)
        q_tot = torch.bmm(hidden, output_weight) + output_bias

        return q_tot.squeeze(1)