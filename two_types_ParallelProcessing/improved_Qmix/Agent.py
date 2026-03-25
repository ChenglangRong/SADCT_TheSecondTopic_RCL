import torch
import torch.nn.functional as F
import numpy as np
import random
from AttentionLayer_Network import AgentQNetwork, MixingNetwork
from PrioritizedReplayBuffer import PrioritizedReplayBuffer


class MAQMixAgent:
    """单个智能体的Q网络封装"""

    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.q_net = AgentQNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = AgentQNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=lr)
        self.device = device
        self.action_dim = action_dim

    def get_q_values(self, state, hidden=None):
        if len(state.size()) == 2:
            state = state.unsqueeze(1)
        return self.q_net(state, hidden)

    def get_target_q_values(self, state, hidden=None):
        if len(state.size()) == 2:
            state = state.unsqueeze(1)
        return self.target_q_net(state, hidden)

    def soft_update(self, tau):
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class MAQMix:
    """改进的MAQMix核心类：加入注意力和优先回放"""

    def __init__(self, state_dim, hidden_dim, action_dims, n_agents, lr, gamma, tau, buffer_size,
                 batch_size, device, hidden_dim_mixing=32, alpha=0.6, beta=0.4):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.beta = beta  # 优先回放的重要性采样参数

        # 初始化各智能体Q网络
        self.agents = [
            MAQMixAgent(state_dim, hidden_dim, action_dims[i], lr, device)
            for i in range(n_agents)
        ]

        # 混合网络与目标混合网络
        self.mixing_net = MixingNetwork(state_dim, n_agents, hidden_dim_mixing).to(device)
        self.target_mixing_net = MixingNetwork(state_dim, n_agents, hidden_dim_mixing).to(device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # 优先经验回放池
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)

        # 联合优化器
        all_params = []
        for agent in self.agents:
            all_params.extend(agent.q_net.parameters())
        all_params.extend(self.mixing_net.parameters())
        self.optimizer = torch.optim.RMSprop(all_params, lr=lr)

    def take_actions(self, state, masks, epsilon=0.05):
        actions = []
        state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)

        for i, agent in enumerate(self.agents):
            q_values, _ = agent.get_q_values(state_tensor)
            mask = torch.tensor(masks[i], dtype=torch.float).to(self.device)
            q_values = q_values * mask - 1e9 * (1 - mask)

            if random.random() < epsilon:
                feasible_actions = torch.where(mask == 1)[0].cpu().numpy()
                action = np.random.choice(feasible_actions)
            else:
                action = torch.argmax(q_values, dim=1).item()

            actions.append(action)
        return actions

    def add_experience(self, state, actions, reward, next_state, done, masks):
        self.replay_buffer.add(state, actions, reward, next_state, done, masks)

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return None

        # 从优先回放池采样
        state, actions, reward, next_state, done, masks, indices, weights = \
            self.replay_buffer.sample(self.batch_size, self.beta)

        # 转换为张量
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        weights = weights.to(self.device)

        # 处理掩码
        agent_masks = []
        for i in range(self.n_agents):
            mask_list = [sample[i] for sample in masks]
            mask_tensor = torch.tensor(mask_list, dtype=torch.float).to(self.device)
            agent_masks.append(mask_tensor)

        # 计算当前全局Q_tot
        current_q_list = []
        for i, agent in enumerate(self.agents):
            q_values, _ = agent.get_q_values(state)
            agent_actions = actions[:, i].unsqueeze(1)
            current_q = torch.gather(q_values, dim=1, index=agent_actions).squeeze(1)
            current_q_list.append(current_q)

        current_q_stack = torch.stack(current_q_list, dim=1)
        current_q_tot = self.mixing_net(current_q_stack, state)

        # 计算目标全局Q_tot
        next_q_list = []
        for i, agent in enumerate(self.agents):
            next_q_values, _ = agent.get_target_q_values(next_state)
            next_q_values = next_q_values * agent_masks[i] - 1e9 * (1 - agent_masks[i])
            max_next_q = torch.max(next_q_values, dim=1)[0]
            next_q_list.append(max_next_q)

        next_q_stack = torch.stack(next_q_list, dim=1)
        next_q_tot = self.target_mixing_net(next_q_stack, next_state)
        target_q_tot = reward + self.gamma * (1 - done) * next_q_tot

        # 计算TD误差和带权重的损失
        td_errors = current_q_tot - target_q_tot.detach()
        loss = (weights * F.mse_loss(current_q_tot, target_q_tot.detach(), reduction='none')).mean()

        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.cpu().detach().numpy())

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        for agent in self.agents:
            agent.soft_update(self.tau)
        for param, target_param in zip(self.mixing_net.parameters(), self.target_mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.item()