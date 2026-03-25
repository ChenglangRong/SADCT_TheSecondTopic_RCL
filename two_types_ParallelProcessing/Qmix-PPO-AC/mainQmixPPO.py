import torch
import random
import numpy as np
from two_types_ParallelProcessing.SADCT_environment2_ParallelProcessing import Environment
from Agent import QmixPPOAC  # 导入Qmix-PPO-AC智能体
import utils
from params import args_QmixPPO  # 导入修正后的参数


class QmixPPOAC_Runner:
    def __init__(self, args, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        self.args = args
        # ========== 核心参数：只保留必要的，删除冗余的GAE ==========
        self.lr_actor = args.lr_actor  # PPO策略学习率
        self.lr_q = args.lr_q  # Qmix局部Q网络学习率（替代lr_critic）
        self.lr_mixer = args.lr_mixer  # Qmix混合网络学习率
        self.gamma = args.gamma  # 折扣因子
        self.clip_epsilon = args.clip_epsilon  # PPO剪切系数（原ppo_clip）
        self.ppo_epochs = args.ppo_epochs  # PPO更新轮数
        self.entropy_coef = args.entropy_coef  # 熵奖励系数
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.hidden_dim_mixing = args.hidden_dim_mixing
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.num_episodes = args.num_episodes
        self.update_freq = args.update_freq

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env

        # 创建环境（修正：传递正确的env_args）
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim
        self.action_dims = self.env.action_dims
        self.n_agents = len(self.action_dims)  # 双智能体

        # 初始化Qmix-PPO-AC智能体（参数严格匹配Agent.py的QmixPPOAC类）
        self.algorithm = QmixPPOAC(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dims=self.action_dims,
            n_agents=self.n_agents,
            lr_qmix=self.lr_mixer,  # Qmix混合网络学习率
            lr_actor=self.lr_actor,  # PPO策略学习率
            lr_critic=self.lr_q,  # 兼容：用Qmix的lr_q替代lr_critic
            gamma=self.gamma,
            tau=self.args.tau,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            clip_epsilon=self.clip_epsilon,  # 传递PPO剪切系数
            device=self.device
        )

        # 创建模型保存目录
        utils.create_directory(args.ckpt_dir,
                               sub_dirs=[f'QmixPPOAC_env_{elect_env_example}'])

        # 数据记录初始化
        self.reward_list = []
        self.makespan_list = []
        self.data = {
            'episode': [], 'reward': [], 'makespan': [], 'fail': []
        }
        self.epsilon = self.epsilon_start

    def run(self):
        # 固定随机种子
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        for i_episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            print(f"\n第{i_episode + 1}回合====================================")

            # ε衰减
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # 收集经验
            while not done:
                # 获取动作掩码（兼容环境的get_mask方法）
                masks = self.env.get_mask() if hasattr(self.env, 'get_mask') else [[1] * self.action_dims[0],
                                                                                   [1] * self.action_dims[1]]
                # 采样动作（返回动作+对数概率）
                actions, log_probs = self.algorithm.take_actions(state, masks, self.epsilon)
                # 环境步进
                next_state, reward, done = self.env.step(actions)

                # 存储经验
                self.algorithm.add_experience(
                    state, actions, log_probs, reward, next_state, done, masks
                )

                state = next_state
                episode_return += reward

                # 定期更新
                if len(self.algorithm.replay_buffer) >= self.batch_size and (i_episode % self.update_freq == 0):
                    q_loss, ppo_loss = self.algorithm.update()
                    if i_episode % 500 == 0:
                        print(f"回合 {i_episode}, Qmix损失: {q_loss:.4f}, PPO损失: {ppo_loss:.4f}")

            # 记录数据
            self.reward_list.append(episode_return)
            self.data['episode'].append(i_episode)
            self.data['reward'].append(episode_return)
            self.data['makespan'].append(
                getattr(self.env, 'now', 0) if hasattr(self.env, 'now') else getattr(self.env, 'env.now', 0))
            self.data['fail'].append(any(getattr(self.env, 'fail_flags', [False])))

            print(f"奖励: {episode_return:.1f}, 完工时间: {self.data['makespan'][-1]}, "
                  f"失败状态: {self.data['fail'][-1]}, ε: {self.epsilon:.3f}")

            # 保存模型
            if (i_episode + 1) % 1000 == 0:
                self.save_models(i_episode)

        return self.reward_list, self.makespan_list, self.data

    def save_models(self, episode):
        """保存Qmix-PPO-AC模型"""
        path = f"{self.args.ckpt_dir}/QmixPPOAC_env_{self.elect_env_example}"
        utils.create_directory(path, [])
        # 保存PPO策略网络
        for i, agent in enumerate(self.algorithm.ppo_agents):
            torch.save(agent.actor.state_dict(), f"{path}/actor_agent_{i}_ep{episode}.pth")
        # 保存Qmix价值网络
        torch.save(self.algorithm.mixing_net.state_dict(), f"{path}/mixing_net_ep{episode}.pth")


# 设置环境参数
def set_env(elect_env_example):
    from params import args_QmixPPO
    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        print("案例选择错误")
        exit(1)
    return QmixPPOAC_Runner(args_QmixPPO, args_env, elect_env_example)


# 主函数
def main():
    elect_env_example = 6  # 选择6号环境案例
    runner = set_env(elect_env_example)
    # 创建结果保存目录
    utils.create_directory(args_QmixPPO.image_dir, [f'QmixPPOAC_env_{elect_env_example}'])
    utils.create_directory(args_QmixPPO.data_dir, [f'QmixPPOAC_env_{elect_env_example}'])
    # 运行训练
    reward_list, makespan_list, data = runner.run()
    # 绘制移动平均奖励曲线
    avg_rewards = utils.moving_average(reward_list, 100)
    utils.plot_method(range(len(avg_rewards)), avg_rewards,
                      'Moving Average Reward', 'reward',
                      f"{args_QmixPPO.image_dir}/QmixPPOAC_env_{elect_env_example}/avg_reward.png")


if __name__ == "__main__":
    main()