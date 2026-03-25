import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from Network import PolicyNetwork, ValueNetwork  # 新增ValueNetwork


class PPOAgent:
    """PPO智能体（单个智能体）"""

    def __init__(self, state_dim, hidden_dim, action_dim, lr_actor, lr_critic,
                 clip_epsilon=0.2, device='cpu'):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)  # 价值网络输出状态价值
        self.old_actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.clip_epsilon = clip_epsilon
        self.device = device
        self.action_dim = action_dim

    def get_action(self, state, masks, deterministic=False):
        """获取动作和概率"""
        # 替换torch.tensor为更安全的转换方式
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().unsqueeze(0).to(self.device, non_blocking=True)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

        if isinstance(masks, torch.Tensor):
            mask_tensor = masks.float().to(self.device, non_blocking=True)
        else:
            mask_tensor = torch.tensor(masks, dtype=torch.float, device=self.device)

        logits = self.actor(state_tensor)
        logits = logits * mask_tensor - 1e9 * (1 - mask_tensor)  # 屏蔽不可行动作
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=1).item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        return action, log_prob

    def evaluate(self, state, action, masks):
        """评估动作概率和状态价值"""
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device, non_blocking=True)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)

        if isinstance(masks, torch.Tensor):
            mask_tensor = masks.float().to(self.device, non_blocking=True)
        else:
            mask_tensor = torch.tensor(masks, dtype=torch.float, device=self.device)

        logits = self.actor(state_tensor)
        logits = logits * mask_tensor - 1e9 * (1 - mask_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state_tensor).squeeze(1)
        return log_prob, value, entropy

    def get_old_log_prob(self, state, action, masks):
        """获取旧策略的动作概率"""
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device, non_blocking=True)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)

        if isinstance(masks, torch.Tensor):
            mask_tensor = masks.float().to(self.device, non_blocking=True)
        else:
            mask_tensor = torch.tensor(masks, dtype=torch.float, device=self.device)

        logits = self.old_actor(state_tensor)
        logits = logits * mask_tensor - 1e9 * (1 - mask_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action)

    def update_old_policy(self):
        """更新旧策略网络"""
        self.old_actor.load_state_dict(self.actor.state_dict())


class PPO:
    """多智能体PPO算法"""

    def __init__(self, state_dim, hidden_dim, action_dims, n_agents=2,
                 lr_actor=3e-5, lr_critic=1e-4, gamma=0.99, gae_lambda=0.95,
                 batch_size=32, clip_epsilon=0.2, ppo_epochs=10, entropy_coef=0.01,
                 device='cpu'):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.device = device

        # 初始化多智能体
        self.agents = [
            PPOAgent(state_dim, hidden_dim, action_dims[i], lr_actor, lr_critic,
                     clip_epsilon, device)
            for i in range(n_agents)
        ]

        # 经验回放池
        self.replay_buffer = []
        self.buffer_size = 200000  # 可配置

    def take_actions(self, state, masks, deterministic=False):
        """获取所有智能体的动作和概率"""
        actions = []
        log_probs = []
        for i, agent in enumerate(self.agents):
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

    def compute_gae(self, rewards, values, next_values, dones):
        """计算GAE优势估计，确保返回的是叶子张量"""
        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage.detach()  # detach避免梯度问题
        returns = advantages + values.detach()  # 使用detach()确保不传播到价值网络
        return advantages, returns

    def update(self):
        """更新PPO算法"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # 开启异常检测（调试用，稳定后可移除）
        torch.autograd.set_detect_anomaly(True)

        # 采样批次数据
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.tensor([b['state'] for b in batch], dtype=torch.float).to(self.device)
        actions = [torch.tensor([b['actions'][i] for b in batch], dtype=torch.long).to(self.device)
                   for i in range(self.n_agents)]
        old_log_probs = [torch.tensor([b['log_probs'][i] for b in batch], dtype=torch.float).to(self.device)
                         for i in range(self.n_agents)]
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float).to(self.device)
        next_states = torch.tensor([b['next_state'] for b in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.float).to(self.device)
        masks = [torch.tensor([b['masks'][i] for b in batch], dtype=torch.float).to(self.device)
                 for i in range(self.n_agents)]

        # 计算价值和优势
        values = []
        next_values = []
        for i, agent in enumerate(self.agents):
            _, val, _ = agent.evaluate(states, actions[i], masks[i])
            values.append(val)
            _, next_val, _ = agent.evaluate(next_states, actions[i], masks[i])
            next_values.append(next_val)

        # 计算GAE优势（使用第一个智能体的价值估计，或平均所有智能体的价值）
        advantages, returns = self.compute_gae(rewards, values[0], next_values[0], dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多智能体PPO更新
        total_actor_loss = 0
        total_critic_loss = 0

        # 在PPO类的update方法中，修改多智能体更新循环
        for epoch in range(self.ppo_epochs):
            # 先计算所有智能体的评估结果，保存计算图
            eval_results = []
            for i, agent in enumerate(self.agents):
                log_probs, values, entropy = agent.evaluate(states, actions[i], masks[i])
                eval_results.append((log_probs, values, entropy))

            # 再分别更新每个智能体，每次更新都重新计算图
            for i, agent in enumerate(self.agents):
                log_probs, values, entropy = eval_results[i]

                # 计算PPO损失
                ratio = torch.exp(log_probs - old_log_probs[i])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - agent.clip_epsilon, 1 + agent.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                # 价值损失
                critic_loss = F.mse_loss(values, returns)

                # 更新网络 - 关键修改：为每个智能体单独创建计算图
                agent.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=False)  # 不保留计算图
                agent.actor_optimizer.step()

                agent.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=False)  # 每个智能体单独 backward
                agent.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

            # 更新旧策略
            for agent in self.agents:
                agent.update_old_policy()

        # 关闭异常检测
        torch.autograd.set_detect_anomaly(False)

        avg_actor_loss = total_actor_loss / (self.ppo_epochs * self.n_agents)
        avg_critic_loss = total_critic_loss / (self.ppo_epochs * self.n_agents)
        return avg_actor_loss, avg_critic_loss