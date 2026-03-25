import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from Network import AgentQNetwork, MixingNetwork, PolicyNetwork


class PPOAgent:
    """PPO策略网络封装（单个智能体）"""

    def __init__(self, state_dim, hidden_dim, action_dim, lr_actor, lr_critic,
                 clip_epsilon=0.2, device='cpu'):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic = AgentQNetwork(state_dim, hidden_dim, 1).to(device)  # 评论者输出状态价值
        self.old_actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.clip_epsilon = clip_epsilon
        self.device = device
        self.action_dim = action_dim

    def get_action(self, state, masks, deterministic=False):
        """获取动作和概率"""
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(masks, dtype=torch.float).to(self.device)

        logits = self.actor(state_tensor)
        logits = logits * mask_tensor - 1e9 * (1 - mask_tensor)  # 屏蔽不可行动作
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=1).item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        return action, log_prob

    def get_old_action_prob(self, state, action, masks):
        """获取旧策略的动作概率"""
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(masks, dtype=torch.float).to(self.device)

        logits = self.old_actor(state_tensor)
        logits = logits * mask_tensor - 1e9 * (1 - mask_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(torch.tensor(action).to(self.device))

    def update_old_policy(self):
        """更新旧策略网络参数"""
        self.old_actor.load_state_dict(self.actor.state_dict())


class QmixPPOAC:
    """Qmix-PPO-AC融合算法（双智能体）"""

    def __init__(self, state_dim, hidden_dim, action_dims, n_agents=2,
                 lr_qmix=1e-4, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, tau=0.01, buffer_size=100000,
                 batch_size=32, clip_epsilon=0.2, device='cpu'):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # PPO策略智能体
        self.ppo_agents = [
            PPOAgent(state_dim, hidden_dim, action_dims[i], lr_actor, lr_critic,
                     clip_epsilon, device)
            for i in range(n_agents)
        ]

        # Qmix价值网络
        self.mixing_net = MixingNetwork(state_dim, n_agents, hidden_dim).to(device)
        self.target_mixing_net = MixingNetwork(state_dim, n_agents, hidden_dim).to(device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # 经验回放池
        self.replay_buffer = []
        self.buffer_size = buffer_size

        # Qmix优化器
        self.q_optimizer = optim.RMSprop(
            list(self.mixing_net.parameters()) +
            [param for agent in self.ppo_agents for param in agent.critic.parameters()],
            lr=lr_qmix
        )

    def take_actions(self, state, masks, deterministic=False):
        """获取双智能体动作和概率"""
        actions = []
        log_probs = []
        for i, agent in enumerate(self.ppo_agents):
            action, log_prob = agent.get_action(state, masks[i], deterministic)
            actions.append(action)
            log_probs.append(log_prob)
        return actions, log_probs

    def add_experience(self, state, actions, log_probs, reward, next_state, done, masks):
        """添加经验到回放池"""
        self.replay_buffer.append({
            'state': state,
            'actions': actions,
            'log_probs': log_probs,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'masks': masks
        })
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def update_qmix(self):
        """更新Qmix价值网络"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 采样批次数据
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.tensor([b['state'] for b in batch], dtype=torch.float).to(self.device)
        actions = torch.tensor([b['actions'] for b in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor([b['next_state'] for b in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.float).view(-1, 1).to(self.device)

        # 处理掩码
        agent_masks = []
        for i in range(self.n_agents):
            mask_list = [b['masks'][i] for b in batch]
            agent_masks.append(torch.tensor(mask_list, dtype=torch.float).to(self.device))

        # 计算当前Q值
        current_q_list = []
        for i, agent in enumerate(self.ppo_agents):
            q_values, _ = agent.critic(states)  # (batch_size, 1)
            current_q_list.append(q_values.squeeze(1))

        current_q_stack = torch.stack(current_q_list, dim=1)
        current_q_tot = self.mixing_net(current_q_stack, states)

        # 计算目标Q值
        next_q_list = []
        for i, agent in enumerate(self.ppo_agents):
            next_q_values, _ = agent.critic(next_states)
            next_q_list.append(next_q_values.squeeze(1))

        next_q_stack = torch.stack(next_q_list, dim=1)
        next_q_tot = self.target_mixing_net(next_q_stack, next_states)
        target_q_tot = rewards + self.gamma * (1 - dones) * next_q_tot

        # 计算Q损失
        q_loss = F.mse_loss(current_q_tot, target_q_tot.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.mixing_net.parameters(), self.target_mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item()

    def update_ppo(self, epochs=10):
        """更新PPO策略网络"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 计算优势值
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.tensor([b['state'] for b in batch], dtype=torch.float).to(self.device)
        actions = [torch.tensor([b['actions'][i] for b in batch], dtype=torch.long).to(self.device)
                   for i in range(self.n_agents)]
        old_log_probs = [torch.tensor([b['log_probs'][i] for b in batch], dtype=torch.float).to(self.device)
                         for i in range(self.n_agents)]
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float).view(-1, 1).to(self.device)

        # 从Qmix获取状态价值
        q_values, _ = self.ppo_agents[0].critic(states)  # 使用任意智能体的评论者
        advantages = rewards - q_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        actor_losses = []
        for epoch in range(epochs):
            for i, agent in enumerate(self.ppo_agents):
                # 计算新策略概率
                logits = agent.actor(states)
                mask_tensor = torch.tensor([b['masks'][i] for b in batch], dtype=torch.float).to(self.device)
                logits = logits * mask_tensor - 1e9 * (1 - mask_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[i])

                # 计算重要性权重
                ratio = torch.exp(new_log_probs - old_log_probs[i])
                surr1 = ratio * advantages.squeeze(1)
                surr2 = torch.clamp(ratio, 1 - agent.clip_epsilon, 1 + agent.clip_epsilon) * advantages.squeeze(1)
                actor_loss = -torch.min(surr1, surr2).mean()

                # 更新演员网络
                agent.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                agent.actor_optimizer.step()
                actor_losses.append(actor_loss.item())

            # 更新评论者网络
            for agent in self.ppo_agents:
                values, _ = agent.critic(states)
                critic_loss = F.mse_loss(values, rewards)
                agent.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                agent.critic_optimizer.step()

        # 更新旧策略
        for agent in self.ppo_agents:
            agent.update_old_policy()

        return sum(actor_losses) / len(actor_losses)

    def update(self):
        """联合更新价值网络和策略网络"""
        q_loss = self.update_qmix()
        ppo_loss = self.update_ppo()
        return q_loss, ppo_loss