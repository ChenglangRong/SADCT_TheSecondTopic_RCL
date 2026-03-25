import torch
import random
import numpy as np
from two_types_ParallelProcessing.SADCT_environment2_ParallelProcessing import Environment
from Agent import MultiAgentMATD3
import utils


class MultiMATD3_Runner:
    def __init__(self, args_MATD3, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        self.actor_lr = args_MATD3.actor_lr
        self.critic_lr = args_MATD3.critic_lr
        self.num_episodes = args_MATD3.num_episodes
        self.hidden_dim = args_MATD3.hidden_dim
        self.gamma = args_MATD3.gamma
        self.tau = args_MATD3.tau
        self.buffer_size = args_MATD3.buffer_size
        self.batch_size = args_MATD3.batch_size
        self.policy_delay = args_MATD3.policy_delay  # 添加MATD3参数
        self.noise_std = args_MATD3.noise_std
        self.noise_clip = args_MATD3.noise_clip

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env

        # 创建环境（保持不变）
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim
        self.action_dims = self.env.action_dims

        # 初始化多智能体MATD3
        self.multi_agent = MultiAgentMATD3(
            self.state_dim, self.hidden_dim, self.action_dims,
            self.actor_lr, self.critic_lr, self.gamma, self.tau,
            self.buffer_size, self.batch_size, self.policy_delay,
            self.noise_std, self.noise_clip, self.device
        )

        # 其余初始化逻辑保持不变
        utils.create_directory(args_MATD3.ckpt_dir,
                               sub_dirs=[f'actor_env_{elect_env_example}',
                                         f'critic_env_{elect_env_example}'])

        self.reward_list = []
        self.makespan_list = []
        self.data = {
            'episode': [], 'reward': [], 'makespan': [], 'fail': []
        }

    # run方法保持不变
    def run(self):
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

                if all(agent.replay_buffer.size() >= self.batch_size for agent in self.multi_agent.agents):
                    self.multi_agent.update()

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

# 后续set_env和main方法保持不变，仅修改引用名称为MATD3
def set_env(elect_env_example):
    from params import args_TD3  # 实际使用时建议重命名为args_MATD3
    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        print("案例选择错误")
        exit(1)
    return MultiMATD3_Runner(args_TD3, args_env, elect_env_example)

def main():
    from params import args_TD3
    elect_env_example = 6
    runner = set_env(elect_env_example)
    utils.create_directory(args_TD3.image_dir, [f'MultiMATD3_env_{elect_env_example}'])
    utils.create_directory(args_TD3.data_dir, [f'MultiMATD3_env_{elect_env_example}'])
    reward_list, makespan_list, data = runner.run()
    avg_rewards = utils.moving_average(reward_list, 100)
    utils.plot_method(range(len(avg_rewards)), avg_rewards,
                      'Moving Average Reward', 'reward',
                      f"{args_TD3.image_dir}/MultiMATD3_env_{elect_env_example}/avg_reward.png")


if __name__ == "__main__":
    main()