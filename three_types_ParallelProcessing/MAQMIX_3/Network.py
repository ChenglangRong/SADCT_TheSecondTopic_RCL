import torch
import torch.nn as nn
import torch.nn.functional as F

# QMix智能体Q网络（DRQN，处理部分可观测性）
class AgentQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)  # 时序建模
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len=1, state_dim) 适配单步观测
        if len(x.size()) == 2:
            x = x.unsqueeze(1)  # 扩展时序维度
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)

        gru_out, hidden = self.gru(x, hidden)
        out = gru_out[:, -1, :]  # 取最后一步输出
        out = F.relu(self.fc1(out))
        q_values = self.fc2(out)  # 输出各动作Q值
        return q_values, hidden


# 超网络：生成混合网络权重（保证非负）
class HyperNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim_mixing=32):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim_mixing = hidden_dim_mixing

        # 生成混合网络各层权重/偏置
        self.fc_hidden_weight = nn.Linear(state_dim, n_agents * hidden_dim_mixing)
        self.fc_hidden_bias = nn.Linear(state_dim, hidden_dim_mixing)
        self.fc_output_weight = nn.Linear(state_dim, hidden_dim_mixing * 1)
        self.fc_output_bias = nn.Sequential(  # 输出偏置用2层网络（QMix论文设定）
            nn.Linear(state_dim, hidden_dim_mixing),
            nn.ReLU(),
            nn.Linear(hidden_dim_mixing, 1)
        )

    def forward(self, s):
        # s: (batch_size, state_dim) 全局状态
        batch_size = s.size(0)

        # 隐藏层权重：(batch_size, n_agents, hidden_dim_mixing)，绝对值保证非负
        hidden_weight = self.fc_hidden_weight(s).view(batch_size, self.n_agents, self.hidden_dim_mixing)
        hidden_weight = torch.abs(hidden_weight)

        # 隐藏层偏置：(batch_size, 1, hidden_dim_mixing)
        hidden_bias = self.fc_hidden_bias(s).view(batch_size, 1, self.hidden_dim_mixing)

        # 输出层权重：(batch_size, hidden_dim_mixing, 1)，绝对值保证非负
        output_weight = self.fc_output_weight(s).view(batch_size, self.hidden_dim_mixing, 1)
        output_weight = torch.abs(output_weight)

        # 输出层偏置：(batch_size, 1, 1)
        output_bias = self.fc_output_bias(s).view(batch_size, 1, 1)

        return hidden_weight, hidden_bias, output_weight, output_bias


# 混合网络：将智能体Q值组合为全局Q_tot
class MixingNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim_mixing=32):
        super().__init__()
        self.n_agents = n_agents
        self.hypernet = HyperNetwork(state_dim, n_agents, hidden_dim_mixing)
        self.elu = nn.ELU()  # QMix论文激活函数

    def forward(self, q_values, s):
        # q_values: (batch_size, n_agents) 各智能体选中动作的Q值
        # s: (batch_size, state_dim) 全局状态
        batch_size = q_values.size(0)
        q_values = q_values.view(batch_size, 1, self.n_agents)  # 维度适配

        # 从超网络获取权重
        hidden_weight, hidden_bias, output_weight, output_bias = self.hypernet(s)

        # 非线性混合计算
        hidden = self.elu(torch.bmm(q_values, hidden_weight) + hidden_bias)
        q_tot = torch.bmm(hidden, output_weight) + output_bias  # 全局Q值

        return q_tot.squeeze(1)  # (batch_size,)