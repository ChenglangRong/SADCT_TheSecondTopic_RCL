import torch
import torch.nn.functional as F
import numpy as np
import random
from Network import AgentQNetwork, MixingNetwork
from ReplayBuffer import ReplayBuffer


class MAQMixAgent:
    """单个智能体的Q网络封装"""

    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.q_net = AgentQNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = AgentQNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 初始化目标网络
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=lr)
        self.device = device
        self.action_dim = action_dim

    def get_q_values(self, state, hidden=None):
        """获取当前状态下所有动作的Q值"""
        if len(state.size()) == 2:
            state = state.unsqueeze(1)  # 适配GRU时序维度
        return self.q_net(state, hidden)

    def get_target_q_values(self, state, hidden=None):
        """获取目标网络的Q值"""
        if len(state.size()) == 2:
            state = state.unsqueeze(1)
        return self.target_q_net(state, hidden)

    def soft_update(self, tau):
        """软更新目标网络"""
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class MAQMix:
    """MAQMix核心类：管理多智能体+混合网络"""

    def __init__(self, state_dim, hidden_dim, action_dims, n_agents, lr, gamma, tau, buffer_size, batch_size, device,
                 hidden_dim_mixing=32):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # 初始化各智能体Q网络
        self.agents = [
            MAQMixAgent(state_dim, hidden_dim, action_dims[i], lr, device)
            for i in range(n_agents)
        ]

        # 混合网络与目标混合网络
        self.mixing_net = MixingNetwork(state_dim, n_agents, hidden_dim_mixing).to(device)
        self.target_mixing_net = MixingNetwork(state_dim, n_agents, hidden_dim_mixing).to(device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # 全局经验回放池（存储联合transition）
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 联合优化器（Q网络+混合网络）
        all_params = []
        for agent in self.agents:
            all_params.extend(agent.q_net.parameters())
        all_params.extend(self.mixing_net.parameters())
        self.optimizer = torch.optim.RMSprop(all_params, lr=lr)  # QMix论文优化器

    def take_actions(self, state, masks, epsilon=0.05):
        """ε-greedy动作选择（分布式执行）"""
        actions = []
        state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)

        for i, agent in enumerate(self.agents):
            q_values, _ = agent.get_q_values(state_tensor)  # (1, action_dim)
            mask = torch.tensor(masks[i], dtype=torch.float).to(self.device)

            # 屏蔽不可行动作（设为极小值）
            q_values = q_values * mask - 1e9 * (1 - mask)

            # ε-greedy选择
            if random.random() < epsilon:
                feasible_actions = torch.where(mask == 1)[0].cpu().numpy()
                action = np.random.choice(feasible_actions)
            else:
                action = torch.argmax(q_values, dim=1).item()

            actions.append(action)
        return actions

    def add_experience(self, state, actions, reward, next_state, done, masks):
        """添加全局经验到回放池"""
        self.replay_buffer.add(state, actions, reward, next_state, done, masks)

    def update(self):
        """集中式训练：最小化TD误差"""
        if self.replay_buffer.size() < self.batch_size:
            return None  # 经验不足时不更新

        # 采样批次数据
        state, actions, reward, next_state, done, masks = self.replay_buffer.sample(self.batch_size)
        # 转换其他数据为张量
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)  # (batch_size, n_agents)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)

        # 关键修改：按智能体拆分掩码，各自转换为二维张量（batch_size, action_dim_i）
        agent_masks = []  # agent_masks[i] 对应智能体i的掩码张量
        for i in range(self.n_agents):
            # 提取批次中所有样本的第i个智能体的掩码
            mask_list = [sample[i] for sample in masks]
            # 转换为张量 (batch_size, action_dim_i)
            mask_tensor = torch.tensor(mask_list, dtype=torch.float).to(self.device)
            agent_masks.append(mask_tensor)

        # 计算当前全局Q_tot
        current_q_list = []
        for i, agent in enumerate(self.agents):
            q_values, _ = agent.get_q_values(state)  # (batch_size, action_dim)
            # 提取选中动作的Q值
            agent_actions = actions[:, i].unsqueeze(1)
            current_q = torch.gather(q_values, dim=1, index=agent_actions).squeeze(1)
            current_q_list.append(current_q)

        current_q_stack = torch.stack(current_q_list, dim=1)  # (batch_size, n_agents)
        current_q_tot = self.mixing_net(current_q_stack, state)  # (batch_size,)

        # 计算目标全局Q_tot
        next_q_list = []
        for i, agent in enumerate(self.agents):
            next_q_values, _ = agent.get_target_q_values(next_state)  # (batch_size, action_dim_i)
            # 应用当前智能体的掩码（agent_masks[i] 形状为 (batch_size, action_dim_i)）
            next_q_values = next_q_values * agent_masks[i] - 1e9 * (1 - agent_masks[i])
            max_next_q = torch.max(next_q_values, dim=1)[0]  # 贪心选择最大Q值
            next_q_list.append(max_next_q)

        next_q_stack = torch.stack(next_q_list, dim=1)  # (batch_size, n_agents)
        next_q_tot = self.target_mixing_net(next_q_stack, next_state)  # (batch_size,)
        target_q_tot = reward + self.gamma * (1 - done) * next_q_tot  # TD目标

        # 计算损失并更新
        loss = F.mse_loss(current_q_tot, target_q_tot.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        for agent in self.agents:
            agent.soft_update(self.tau)
        for param, target_param in zip(self.mixing_net.parameters(), self.target_mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.item()