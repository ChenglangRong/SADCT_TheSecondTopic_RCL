import torch
import torch.nn.functional as F
import numpy as np
from Network import DDPGActor
from Network import DDPGCritic
from ReplayBuffer import ReplayBuffer

class DDPGAgent:
    """单个智能体的DDPG实现"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, device):
        # 策略网络
        self.actor = DDPGActor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = DDPGActor(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = DDPGCritic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = DDPGCritic(state_dim, hidden_dim, action_dim).to(device)

        # 初始化目标网络参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 超参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新系数
        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)

    def take_action(self, state, mask):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state).cpu().detach().numpy()[0]

        # 应用动作掩码
        masked_probs = probs * np.array(mask, dtype=np.float32)

        # 计算概率和，处理总和为 0 的情况
        sum_probs = masked_probs.sum()
        if sum_probs == 0:
            # 没有有效动作时的处理（根据实际场景调整，此处示例返回默认动作 0）
            # 可选：打印警告以排查环境掩码问题
            print(f"警告：动作掩码全为 0，无有效动作！返回默认动作 0")
            return 0  # 或其他合理的默认动作索引

        # 正常归一化
        masked_probs = masked_probs / sum_probs

        # 按概率采样动作
        action = np.random.choice(self.action_dim, p=masked_probs)
        return action

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return None, None

        # 采样
        state, action, reward, next_state, done, mask = self.replay_buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)

        # 目标动作选择
        next_probs = self.target_actor(next_state)
        masked_next_probs = next_probs * mask
        sum_next_probs = masked_next_probs.sum(dim=1, keepdim=True)
        sum_next_probs = torch.where(sum_next_probs == 0, torch.ones_like(sum_next_probs), sum_next_probs)
        masked_next_probs = masked_next_probs / sum_next_probs
        next_action = torch.argmax(masked_next_probs, dim=1).view(-1, 1)

        # 目标Q值计算
        next_q_values = self.target_critic(next_state, F.one_hot(next_action.squeeze(), self.action_dim).float())
        q_targets = reward + self.gamma * (1 - done) * next_q_values

        # 当前Q值
        action_onehot = F.one_hot(action, self.action_dim).float()
        action_onehot = action_onehot * mask
        q_values = self.critic(state, action_onehot)

        # Critic更新
        critic_loss = F.mse_loss(q_values, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新
        actor_probs = self.actor(state)
        masked_actor_probs = actor_probs * mask
        sum_probs = masked_actor_probs.sum(dim=1, keepdim=True)
        sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
        masked_actor_probs = masked_actor_probs / sum_probs

        # 熵正则化
        entropy = -torch.sum(masked_actor_probs * torch.log(masked_actor_probs + 1e-8), dim=1).mean()
        q_values_actor = self.critic(state, masked_actor_probs)
        actor_loss = -torch.mean(q_values_actor) - 0.001 * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()


class MultiAgentDDPG:
    """双智能体协调器"""

    def __init__(self, state_dim, hidden_dim, action_dims, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, device):
        # 初始化两个智能体（分别控制robot1和robot2）
        self.agents = [
            DDPGAgent(
                state_dim, hidden_dim, action_dims[0],  # robot1动作维度
                actor_lr, critic_lr, gamma, tau, buffer_size, batch_size, device
            ),
            DDPGAgent(
                state_dim, hidden_dim, action_dims[1],  # robot2动作维度
                actor_lr, critic_lr, gamma, tau, buffer_size, batch_size, device
            )
        ]
        self.device = device

    def take_actions(self, state, masks):
        """获取两个智能体的动作"""
        return [
            self.agents[0].take_action(state, masks[0]),  # robot1动作
            self.agents[1].take_action(state, masks[1])  # robot2动作
        ]

    def add_experiences(self, state, actions, reward, next_state, done, masks):
        """向两个智能体的经验池添加经验"""
        # 共享相同的状态和奖励，不同的动作和掩码
        self.agents[0].replay_buffer.add(state, actions[0], reward, next_state, done, masks[0])
        self.agents[1].replay_buffer.add(state, actions[1], reward, next_state, done, masks[1])

    def update(self):
        """更新两个智能体的网络"""
        losses = []
        for agent in self.agents:
            critic_loss, actor_loss = agent.update()
            losses.append((critic_loss, actor_loss))
        return losses

    def save_models(self, path):
        """保存模型参数"""
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{path}/actor_agent_{i}.pth")
            torch.save(agent.critic.state_dict(), f"{path}/critic_agent_{i}.pth")

    def load_models(self, path):
        """加载模型参数"""
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{path}/actor_agent_{i}.pth"))
            agent.critic.load_state_dict(torch.load(f"{path}/critic_agent_{i}.pth"))
            agent.target_actor.load_state_dict(agent.actor.state_dict())
            agent.target_critic.load_state_dict(agent.critic.state_dict())