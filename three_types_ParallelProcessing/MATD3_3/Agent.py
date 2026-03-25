import torch
import torch.nn.functional as F
import numpy as np
from Network import TD3Actor
from Network import TD3Critic
from ReplayBuffer import ReplayBuffer


class MATD3Agent:
    """单个智能体的MATD3实现"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, policy_delay,
                 noise_std, noise_clip, device):
        # 策略网络
        self.actor = TD3Actor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = TD3Actor(state_dim, hidden_dim, action_dim).to(device)

        # 双Critic网络（MATD3核心改进）
        self.critic1 = TD3Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = TD3Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = TD3Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = TD3Critic(state_dim, hidden_dim, action_dim).to(device)

        # 初始化目标网络参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay  # 延迟更新策略网络
        self.noise_std = noise_std  # 目标动作噪声
        self.noise_clip = noise_clip  # 噪声剪辑
        self.device = device
        self.action_dim = action_dim
        self.update_count = 0  # 记录更新次数，用于延迟更新

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)

    def take_action(self, state, mask):
        # 动作选择逻辑与DDPG相同
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state).cpu().detach().numpy()[0]
        masked_probs = probs * np.array(mask, dtype=np.float32)
        masked_probs = masked_probs / masked_probs.sum() if masked_probs.sum() > 0 else np.ones_like(
            masked_probs) / len(masked_probs)
        action = np.random.choice(self.action_dim, p=masked_probs)
        return action

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return None, None, None

        # 采样
        state, action, reward, next_state, done, mask = self.replay_buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)

        # 目标动作添加噪声（MATD3核心改进：目标策略平滑）
        next_probs = self.target_actor(next_state)
        # 添加噪声
        noise = torch.randn_like(next_probs) * self.noise_std
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        noisy_next_probs = next_probs + noise
        # 应用掩码并归一化
        masked_next_probs = noisy_next_probs * mask
        sum_next_probs = masked_next_probs.sum(dim=1, keepdim=True)
        sum_next_probs = torch.where(sum_next_probs == 0, torch.ones_like(sum_next_probs), sum_next_probs)
        masked_next_probs = masked_next_probs / sum_next_probs
        next_action = torch.argmax(masked_next_probs, dim=1).view(-1, 1)

        # 计算双目标Q值并取最小值（MATD3核心改进）
        next_q1 = self.target_critic1(next_state, F.one_hot(next_action.squeeze(), self.action_dim).float())
        next_q2 = self.target_critic2(next_state, F.one_hot(next_action.squeeze(), self.action_dim).float())
        next_q = torch.min(next_q1, next_q2)
        q_targets = reward + self.gamma * (1 - done) * next_q

        # 当前Q值
        action_onehot = F.one_hot(action, self.action_dim).float()
        action_onehot = action_onehot * mask
        q1 = self.critic1(state, action_onehot)
        q2 = self.critic2(state, action_onehot)

        # 更新双Critic网络
        critic1_loss = F.mse_loss(q1, q_targets.detach())
        critic2_loss = F.mse_loss(q2, q_targets.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 延迟更新Actor网络（MATD3核心改进）
        actor_loss = None
        self.update_count += 1
        if self.update_count % self.policy_delay == 0:
            actor_probs = self.actor(state)
            masked_actor_probs = actor_probs * mask
            sum_probs = masked_actor_probs.sum(dim=1, keepdim=True)
            sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
            masked_actor_probs = masked_actor_probs / sum_probs

            # 熵正则化
            entropy = -torch.sum(masked_actor_probs * torch.log(masked_actor_probs + 1e-8), dim=1).mean()
            q_values_actor = self.critic1(state, masked_actor_probs)  # 仅用第一个critic更新actor
            actor_loss = -torch.mean(q_values_actor) - 0.001 * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新所有目标网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item() if actor_loss is not None else None


class MultiAgentMATD3:
    """多智能体MATD3协调器"""

    def __init__(self, state_dim, hidden_dim, action_dims, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, policy_delay,
                 noise_std, noise_clip, device):
        # 初始化智能体
        self.agents = [
            MATD3Agent(
                state_dim, hidden_dim, action_dims[0],
                actor_lr, critic_lr, gamma, tau, buffer_size, batch_size,
                policy_delay, noise_std, noise_clip, device
            ),
            MATD3Agent(
                state_dim, hidden_dim, action_dims[1],
                actor_lr, critic_lr, gamma, tau, buffer_size, batch_size,
                policy_delay, noise_std, noise_clip, device
            )
        ]
        self.device = device

    # 以下方法与MADDPG类似，保持接口一致
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
            c1_loss, c2_loss, a_loss = agent.update()
            losses.append((c1_loss, c2_loss, a_loss))
        return losses

    def save_models(self, path):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{path}/actor_agent_{i}.pth")
            torch.save(agent.critic1.state_dict(), f"{path}/critic1_agent_{i}.pth")
            torch.save(agent.critic2.state_dict(), f"{path}/critic2_agent_{i}.pth")

    def load_models(self, path):
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{path}/actor_agent_{i}.pth"))
            agent.critic1.load_state_dict(torch.load(f"{path}/critic1_agent_{i}.pth"))
            agent.critic2.load_state_dict(torch.load(f"{path}/critic2_agent_{i}.pth"))
            agent.target_actor.load_state_dict(agent.actor.state_dict())
            agent.target_critic1.load_state_dict(agent.critic1.state_dict())
            agent.target_critic2.load_state_dict(agent.critic2.state_dict())