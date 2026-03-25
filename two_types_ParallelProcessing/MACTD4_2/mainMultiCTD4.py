import torch
import random
import numpy as np
from two_types_ParallelProcessing.SADCT_environment2_ParallelProcessing import Environment
from Agent import MultiAgentMACTD4
import utils

class MultiMACTD4_Runner:
    def __init__(self, args_CTD4, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        self.actor_lr = args_CTD4.actor_lr
        self.critic_lr = args_CTD4.critic_lr
        self.num_episodes = args_CTD4.num_episodes
        self.hidden_dim = args_CTD4.hidden_dim
        self.gamma = args_CTD4.gamma
        self.tau = args_CTD4.tau
        self.buffer_size = args_CTD4.buffer_size
        self.batch_size = args_CTD4.batch_size
        self.policy_delay = args_CTD4.policy_delay
        self.noise_std = args_CTD4.noise_std
        self.noise_clip = args_CTD4.noise_clip
        self.n_critics = args_CTD4.n_critics
        self.noise_decay_rate = args_CTD4.noise_decay_rate
        self.min_noise_std = args_CTD4.min_noise_std
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env

        # 创建环境（保持不变）
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim
        self.action_dims = self.env.action_dims

        # 初始化多智能体MACTD4
        self.multi_agent = MultiAgentMACTD4(
            self.state_dim, self.hidden_dim, self.action_dims,
            self.actor_lr, self.critic_lr, self.gamma, self.tau,
            self.buffer_size, self.batch_size, self.policy_delay,
            self.noise_std, self.noise_clip, self.device,
            self.n_critics, self.noise_decay_rate, self.min_noise_std
        )

        # 目录创建（保持不变）
        utils.create_directory(args_CTD4.ckpt_dir,
                               sub_dirs=[f'actor_env_{elect_env_example}',
                                         f'critic_env_{elect_env_example}'])
        self.reward_list = []
        self.makespan_list = []
        self.data = {
            'episode': [], 'reward': [], 'makespan': [], 'fail': []
        }

    def run(self):
        """运行逻辑（保持原有流程不变）"""
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        for i_episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            print(f"\n第{i_episode + 1}回合====================================")
            while not done:
                masks = self.env.get_mask()
                actions = self.multi_agent.take_actions(state, masks)
                next_state, reward, done = self.env.step(actions)
                self.multi_agent.add_experiences(state, actions, reward, next_state, done, masks)
                state = next_state
                episode_return += reward
                # 所有智能体回放池都满足批量大小后才更新
                if all(agent.replay_buffer.size() >= self.batch_size for agent in self.multi_agent.agents):
                    self.multi_agent.update()
            # 记录数据（保持不变）
            self.reward_list.append(episode_return)
            self.data['episode'].append(i_episode)
            self.data['reward'].append(episode_return)
            self.data['makespan'].append(self.env.env.now)
            self.data['fail'].append(any(self.env.fail_flags))
            print(f"奖励: {episode_return}, 完工时间: {self.env.env.now}, "
                  f"失败状态: {any(self.env.fail_flags)}")
            if not any(self.env.fail_flags):
                self.makespan_list.append(self.env.env.now)
        return self.reward_list, self.makespan_list, self.data


def set_env(elect_env_example):
    """环境配置（保持不变）"""
    from params import args_CTD4
    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        print("案例选择错误")
        exit(1)
    return MultiMACTD4_Runner(args_CTD4, args_env, elect_env_example)


def main():
    """主函数（保持原有逻辑，修改引用名称）"""
    from params import args_CTD4
    elect_env_example = 6
    runner = set_env(elect_env_example)
    # 创建目录
    utils.create_directory(args_CTD4.image_dir, [f'MultiMACTD4_env_{elect_env_example}'])
    utils.create_directory(args_CTD4.data_dir, [f'MultiMACTD4_env_{elect_env_example}'])
    # 运行训练
    reward_list, makespan_list, data = runner.run()
    avg_rewards = utils.moving_average(reward_list, 100)
    utils.plot_method(range(len(avg_rewards)), avg_rewards,
                      'Moving Average Reward', 'reward',
                      f"{args_CTD4.image_dir}/MultiMACTD4_env_{elect_env_example}/avg_reward.png")


if __name__ == "__main__":
    main()