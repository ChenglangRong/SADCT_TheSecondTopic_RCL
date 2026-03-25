import torch
import random
import numpy as np
from two_types_ParallelProcessing.SADCT_environment2_ParallelProcessing import Environment
from Agent import MAQMix
import utils


class MAQMix_Runner:
    def __init__(self, args_MAQMix, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        self.args = args_MAQMix
        self.lr = args_MAQMix.lr
        self.gamma = args_MAQMix.gamma
        self.tau = args_MAQMix.tau
        self.buffer_size = args_MAQMix.buffer_size
        self.batch_size = args_MAQMix.batch_size
        self.hidden_dim = args_MAQMix.hidden_dim
        self.hidden_dim_mixing = args_MAQMix.hidden_dim_mixing
        self.epsilon_start = args_MAQMix.epsilon_start
        self.epsilon_end = args_MAQMix.epsilon_end
        self.epsilon_decay = args_MAQMix.epsilon_decay
        self.num_episodes = args_MAQMix.num_episodes

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env

        # 创建环境
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim
        self.action_dims = self.env.action_dims
        self.n_agents = len(self.action_dims)

        # 初始化MAQMix智能体
        self.maqmix = MAQMix(
            self.state_dim, self.hidden_dim, self.action_dims, self.n_agents,
            self.lr, self.gamma, self.tau, self.buffer_size,
            self.batch_size, self.device, self.hidden_dim_mixing
        )

        # 创建模型保存目录
        utils.create_directory(args_MAQMix.ckpt_dir,
                               sub_dirs=[f'MAQMix_env_{elect_env_example}'])

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

            while not done:
                masks = self.env.get_mask()
                actions = self.maqmix.take_actions(state, masks, self.epsilon)
                next_state, reward, done = self.env.step(actions)
                # 存储经验（掩码转为tuple适配回放池）
                masks_tuple = (tuple(masks[0]), tuple(masks[1]))
                self.maqmix.add_experience(state, actions, reward, next_state, done, masks_tuple)
                state = next_state
                episode_return += reward

                # 经验池大小足够时进行更新
                if self.maqmix.replay_buffer.size() >= self.batch_size:
                    loss = self.maqmix.update()
                    if i_episode % 500 == 0:
                        print(f"回合 {i_episode}, 损失: {loss:.4f}")

            # 记录数据
            self.reward_list.append(episode_return)
            self.data['episode'].append(i_episode)
            self.data['reward'].append(episode_return)
            self.data['makespan'].append(self.env.env.now)
            self.data['fail'].append(any(self.env.fail_flags))

            print(f"奖励: {episode_return:.1f}, 完工时间: {self.env.env.now}, "
                  f"失败状态: {any(self.env.fail_flags)}, ε: {self.epsilon:.3f}")

            # 保存模型
            if (i_episode + 1) % 1000 == 0:
                self.        save_models(i_episode)

            # 记录成功案例的完工时间
            if not any(self.env.fail_flags):
                self.makespan_list.append(self.env.env.now)

        return self.reward_list, self.makespan_list, self.data

    def save_models(self, episode):
        """保存模型参数"""
        path = f"{self.args.ckpt_dir}/MAQMix_env_{self.elect_env_example}"
        utils.create_directory(path,[])
        # 保存各智能体Q网络
        for i, agent in enumerate(self.maqmix.agents):
            torch.save(agent.q_net.state_dict(), f"{path}/q_net_agent_{i}_ep{episode}.pth")
            torch.save(agent.target_q_net.state_dict(), f"{path}/target_q_net_agent_{i}_ep{episode}.pth")
        # 保存混合网络
        torch.save(self.maqmix.mixing_net.state_dict(), f"{path}/mixing_net_ep{episode}.pth")


def set_env(elect_env_example):
    from params import args_MAQMix
    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        print("案例选择错误")
        exit(1)
    return MAQMix_Runner(args_MAQMix, args_env, elect_env_example)


def main():
    from params import args_MAQMix
    elect_env_example = 6
    runner = set_env(elect_env_example)
    # 创建结果保存目录
    utils.create_directory(args_MAQMix.image_dir, [f'MAQMix_env_{elect_env_example}'])
    utils.create_directory(args_MAQMix.data_dir, [f'MAQMix_env_{elect_env_example}'])
    # 运行训练
    reward_list, makespan_list, data = runner.run()
    # 绘制移动平均奖励曲线
    avg_rewards = utils.moving_average(reward_list, 100)
    utils.plot_method(range(len(avg_rewards)), avg_rewards,
                      'Moving Average Reward', 'reward',
                      f"{args_MAQMix.image_dir}/MAQMix_env_{elect_env_example}/avg_reward.png")


if __name__ == "__main__":
    main()