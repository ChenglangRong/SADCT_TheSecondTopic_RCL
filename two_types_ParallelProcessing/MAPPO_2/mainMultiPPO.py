import torch
import random
import numpy as np
from two_types_ParallelProcessing.SADCT_environment2_ParallelProcessing import Environment
from Agent import PPO
import utils
from params import args_PPO


class PPO_Runner:
    def __init__(self, args, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        self.args = args

        # 初始化参数
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_epsilon = args.clip_epsilon
        self.ppo_epochs = args.ppo_epochs
        self.entropy_coef = args.entropy_coef
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.num_episodes = args.num_episodes
        self.update_freq = args.update_freq

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env

        # 创建环境
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim
        self.action_dims = self.env.action_dims
        self.n_agents = len(self.action_dims)

        # 初始化PPO智能体
        self.algorithm = PPO(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dims=self.action_dims,
            n_agents=self.n_agents,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            batch_size=self.batch_size,
            clip_epsilon=self.clip_epsilon,
            ppo_epochs=self.ppo_epochs,
            entropy_coef=self.entropy_coef,
            device=self.device
        )

        # 创建模型保存目录
        utils.create_directory(args.ckpt_dir,
                               sub_dirs=[f'PPO_env_{elect_env_example}'])

        # 数据记录
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
                masks = self.env.get_mask() if hasattr(self.env, 'get_mask') else [[1] * self.action_dims[0],
                                                                                   [1] * self.action_dims[1]]
                actions, log_probs = self.algorithm.take_actions(state, masks, self.epsilon < random.random())
                next_state, reward, done = self.env.step(actions)

                # 存储经验
                self.algorithm.add_experience(
                    state, actions, log_probs, reward, next_state, done, masks
                )

                state = next_state
                episode_return += reward

                # 定期更新
                if len(self.algorithm.replay_buffer) >= self.batch_size and (i_episode % self.update_freq == 0):
                    actor_loss, critic_loss = self.algorithm.update()
                    if i_episode % 500 == 0:
                        print(f"回合 {i_episode}, 策略损失: {actor_loss:.4f}, 价值损失: {critic_loss:.4f}")

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
        """保存PPO模型"""
        path = f"{self.args.ckpt_dir}/PPO_env_{self.elect_env_example}"
        utils.create_directory(path, [])
        for i, agent in enumerate(self.algorithm.agents):
            torch.save(agent.actor.state_dict(), f"{path}/actor_agent_{i}_ep{episode}.pth")
            torch.save(agent.critic.state_dict(), f"{path}/critic_agent_{i}_ep{episode}.pth")

def set_env(elect_env_example):
    from params import args_PPO
    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        print("案例选择错误")
        exit(1)
    return PPO_Runner(args_PPO, args_env, elect_env_example)


def main():
    from params import args_PPO
    elect_env_example = 6
    runner = set_env(elect_env_example)
    # 创建结果保存目录
    utils.create_directory(args_PPO.image_dir, [f'MAPPO_env_{elect_env_example}'])
    utils.create_directory(args_PPO.data_dir, [f'MAPPO_env_{elect_env_example}'])
    # 运行训练
    reward_list, makespan_list, data = runner.run()
    # 绘制移动平均奖励曲线
    avg_rewards = utils.moving_average(reward_list, 100)
    utils.plot_method(range(len(avg_rewards)), avg_rewards,
                      'Moving Average Reward', 'reward',
                      f"{args_PPO.image_dir}/MAPPO_env_{elect_env_example}/avg_reward.png")


if __name__ == "__main__":
    main()