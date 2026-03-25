import torch
import torch.nn.functional as F
import numpy as np
from Network import CTD4Actor, CTD4Critic
from ReplayBuffer import ReplayBuffer


class MACTD4Agent:
    """单个智能体的MACTD4实现（CTD4多智能体版本）"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, policy_delay,
                 noise_std, noise_clip, device, n_critics=3, noise_decay_rate=0.999, min_noise_std=0.01):
        # 策略网络（保持TD3的确定性策略）
        self.actor = CTD4Actor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = CTD4Actor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # CTD4核心：多Critic网络（默认3个）及目标网络
        self.n_critics = n_critics
        self.critics = [CTD4Critic(state_dim, hidden_dim, action_dim).to(device) for _ in range(n_critics)]
        self.target_critics = [CTD4Critic(state_dim, hidden_dim, action_dim).to(device) for _ in range(n_critics)]
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        # 超参数（添加CTD4的噪声衰减）
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.initial_noise_std = noise_std
        self.noise_clip = noise_clip
        self.noise_decay_rate = noise_decay_rate
        self.min_noise_std = min_noise_std
        self.device = device
        self.action_dim = action_dim
        self.update_count = 0

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)

    def kalman_fusion(self, mus, sigmas):
        """卡尔曼融合多Critic的μ和σ（论文核心公式）"""
        # 初始化融合参数（第一个Critic为初始值）
        fused_mu = mus[0]
        fused_sigma_sq = sigmas[0] ** 2

        # 逐次融合后续Critic
        for mu, sigma in zip(mus[1:], sigmas[1:]):
            sigma_sq = sigma ** 2
            # 卡尔曼增益
            k = fused_sigma_sq / (fused_sigma_sq + sigma_sq)
            # 更新融合参数
            fused_mu = fused_mu + k * (mu - fused_mu)
            fused_sigma_sq = (1 - k) * fused_sigma_sq
        return fused_mu, torch.sqrt(fused_sigma_sq)

    """动作选择（仅对允许动作添加噪声衰减）"""

    def take_action(self, state, mask):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state).cpu().detach().numpy()[0]
        masked_probs = probs * np.array(mask, dtype=np.float32)  # 掩码过滤禁止动作
        allowed_indices = np.where(np.array(mask) == 1)[0]

        if len(allowed_indices) == 0:
            return np.random.choice(self.action_dim)  # 理论上不应触发

        # 提取并处理允许动作的概率
        allowed_probs = masked_probs[allowed_indices]
        allowed_probs = allowed_probs / allowed_probs.sum()  # 初始归一化

        # 探索噪声衰减
        current_noise_std = max(self.noise_std * (self.noise_decay_rate ** self.update_count), self.min_noise_std)

        # 添加噪声并裁剪
        noisy_allowed_probs = allowed_probs + np.random.normal(0, current_noise_std, size=len(allowed_indices))
        noisy_allowed_probs = np.clip(noisy_allowed_probs, 0, 1)  # 确保概率在[0,1]

        # 关键修复：处理总和为0的情况（避免NaN）
        probs_sum = noisy_allowed_probs.sum()
        if probs_sum < 1e-9:  # 总和接近0时，使用均匀分布
            noisy_allowed_probs = np.ones_like(noisy_allowed_probs) / len(noisy_allowed_probs)
        else:
            noisy_allowed_probs = noisy_allowed_probs / probs_sum  # 正常归一化

        # 构建最终概率分布
        final_probs = np.zeros_like(masked_probs)
        final_probs[allowed_indices] = noisy_allowed_probs

        # 选择动作
        action = np.random.choice(self.action_dim, p=final_probs)
        return action

    def soft_update(self, net, target_net):
        """软更新（保持TD3逻辑）"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def kl_loss(self, mu1, sigma1, mu2, sigma2):
        """两个正态分布的KL散度（论文损失函数）"""
        sigma1_sq = sigma1 ** 2
        sigma2_sq = sigma2 ** 2
        kl = torch.log(sigma2 / sigma1) + (sigma1_sq + (mu1 - mu2) ** 2) / (2 * sigma2_sq) - 0.5
        return kl.mean()

    def update(self):
        """CTD4核心更新逻辑"""
        if self.replay_buffer.size() < self.batch_size:
            return [None] * self.n_critics, None

        # 1. 采样数据
        state, action, reward, next_state, done, mask = self.replay_buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)

        # 2. 目标动作生成（保留TD3的目标策略平滑）
        next_probs = self.target_actor(next_state)
        current_noise_std = max(self.noise_std * (self.noise_decay_rate ** self.update_count), self.min_noise_std)
        noise = torch.randn_like(next_probs) * current_noise_std
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        noisy_next_probs = next_probs + noise

        # 掩码和归一化
        masked_next_probs = noisy_next_probs * mask
        sum_next_probs = masked_next_probs.sum(dim=1, keepdim=True)
        sum_next_probs = torch.where(sum_next_probs == 0, torch.ones_like(sum_next_probs), sum_next_probs)
        masked_next_probs = masked_next_probs / sum_next_probs
        next_action = torch.argmax(masked_next_probs, dim=1).view(-1, 1)
        next_action_onehot = F.one_hot(next_action.squeeze(), self.action_dim).float()

        # 3. 计算目标分布（CTD4线性高斯变换）
        next_mus = []
        next_sigmas = []
        for target_critic in self.target_critics:
            mu, sigma = target_critic(next_state, next_action_onehot)
            next_mus.append(mu)
            next_sigmas.append(sigma)
        # 卡尔曼融合目标Critic
        fused_next_mu, fused_next_sigma = self.kalman_fusion(next_mus, next_sigmas)
        # 目标分布参数（论文公式5、6）
        target_mu = self.gamma * fused_next_mu * (1 - done) + reward
        target_sigma = self.gamma * fused_next_sigma * (1 - done)

        # 4. 更新多Critic网络（KL散度损失）
        critic_losses = []
        current_mus = []
        current_sigmas = []
        for i, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers)):
            mu, sigma = critic(state, F.one_hot(action, self.action_dim).float() * mask)
            current_mus.append(mu.detach())
            current_sigmas.append(sigma.detach())
            # KL散度损失
            loss = self.kl_loss(mu, sigma, target_mu.detach(), target_sigma.detach())
            critic_losses.append(loss.item())
            # 优化
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # 始终保留计算图，直到Actor更新完成
            optimizer.step()

        # 5. 延迟更新Actor网络（保留TD3逻辑）
        actor_loss = None
        self.update_count += 1
        if self.update_count % self.policy_delay == 0:
            # 融合当前Critic的输出
            fused_mu, _ = self.kalman_fusion(current_mus, current_sigmas)
            # Actor损失（最大化融合后μ + 熵正则）
            actor_probs = self.actor(state)
            masked_actor_probs = actor_probs * mask
            sum_probs = masked_actor_probs.sum(dim=1, keepdim=True)
            sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
            masked_actor_probs = masked_actor_probs / sum_probs
            entropy = -torch.sum(masked_actor_probs * torch.log(masked_actor_probs + 1e-8), dim=1).mean()
            actor_loss = -torch.mean(fused_mu) - 0.001 * entropy
            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # 软更新所有目标网络
            self.soft_update(self.actor, self.target_actor)
            for critic, target_critic in zip(self.critics, self.target_critics):
                self.soft_update(critic, target_critic)

        return critic_losses, actor_loss.item() if actor_loss is not None else None


class MultiAgentMACTD4:
    """多智能体MACTD4协调器（保持原有接口）"""

    def __init__(self, state_dim, hidden_dim, action_dims, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, policy_delay,
                 noise_std, noise_clip, device, n_critics=3, noise_decay_rate=0.999, min_noise_std=0.01):
        # 初始化两个智能体（与原有MATD3一致）
        self.agents = [
            MACTD4Agent(
                state_dim, hidden_dim, action_dims[0],
                actor_lr, critic_lr, gamma, tau, buffer_size, batch_size,
                policy_delay, noise_std, noise_clip, device, n_critics, noise_decay_rate, min_noise_std
            ),
            MACTD4Agent(
                state_dim, hidden_dim, action_dims[1],
                actor_lr, critic_lr, gamma, tau, buffer_size, batch_size,
                policy_delay, noise_std, noise_clip, device, n_critics, noise_decay_rate, min_noise_std
            )
        ]
        self.device = device

    def take_actions(self, state, masks):
        return [
            self.agents[0].take_action(state, masks[0]),
            self.agents[1].take_action(state, masks[1])
        ]

    def add_experiences(self, state, actions, reward, next_state, done, masks):
        self.agents[0].replay_buffer.add(state, actions[0], reward, next_state, done, masks[0])
        self.agents[1].replay_buffer.add(state, actions[1], reward, next_state, done, masks[1])

    def update(self):
        losses = []
        for agent in self.agents:
            critic_losses, actor_loss = agent.update()
            losses.append((critic_losses, actor_loss))
        return losses

    def save_models(self, path):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{path}/actor_agent_{i}.pth")
            for j, critic in enumerate(agent.critics):
                torch.save(critic.state_dict(), f"{path}/critic_{j}_agent_{i}.pth")

    def load_models(self, path):
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{path}/actor_agent_{i}.pth"))
            agent.target_actor.load_state_dict(agent.actor.state_dict())
            for j, (critic, target_critic) in enumerate(zip(agent.critics, agent.target_critics)):
                critic.load_state_dict(torch.load(f"{path}/critic_{j}_agent_{i}.pth"))
                target_critic.load_state_dict(critic.state_dict())