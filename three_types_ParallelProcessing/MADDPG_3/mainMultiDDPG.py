import torch
import random
import numpy as np
from three_types_ParallelProcessing.SADCT_environment3_ParallelProcessing import Environment
from Agent import MultiAgentDDPG
import utils


class MultiDDPG_Runner:
    def __init__(self, args_DDPG, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        self.actor_lr = args_DDPG.actor_lr
        self.critic_lr = args_DDPG.critic_lr
        self.num_episodes = args_DDPG.num_episodes
        self.hidden_dim = args_DDPG.hidden_dim
        self.gamma = args_DDPG.gamma
        self.tau = args_DDPG.tau
        self.buffer_size = args_DDPG.buffer_size
        self.batch_size = args_DDPG.batch_size

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env

        # 创建环境
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim
        self.action_dims = self.env.action_dims  # [robot1动作数, robot2动作数]

        # 初始化双智能体
        self.multi_agent = MultiAgentDDPG(
            self.state_dim, self.hidden_dim, self.action_dims,
            self.actor_lr, self.critic_lr, self.gamma, self.tau,
            self.buffer_size, self.batch_size, self.device
        )

        # 创建保存目录
        utils.create_directory(args_DDPG.ckpt_dir,
                               sub_dirs=[f'actor_env_{elect_env_example}',
                                         f'critic_env_{elect_env_example}'])

        # 记录数据
        self.reward_list = []
        self.makespan_list = []
        self.data = {
            'episode': [], 'reward': [], 'makespan': [], 'fail': []
        }

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
                # 获取两个机器人的动作掩码
                masks = self.env.get_mask()  # 应返回[masks1, masks2]

                # 双智能体选择动作
                actions = self.multi_agent.take_actions(state, masks)

                # 执行动作
                next_state, reward, done = self.env.step(actions)

                # 存储经验
                self.multi_agent.add_experiences(state, actions, reward, next_state, done, masks)

                # 更新状态和累积奖励
                state = next_state
                episode_return += reward

                # 更新智能体网络
                if all(agent.replay_buffer.size() >= self.batch_size for agent in self.multi_agent.agents):
                    self.multi_agent.update()

            # 记录数据
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
    from params import args_DDPG  # 需要在params.py中添加DDPG参数

    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        print("案例选择错误")
        exit(1)
    return MultiDDPG_Runner(args_DDPG, args_env, elect_env_example)

def main():
    from params import args_DDPG
    elect_env_example = 6  # 选择案例
    runner = set_env(elect_env_example)
    # 创建保存目录
    utils.create_directory(args_DDPG.image_dir, [f'MultiDDPG_env_{elect_env_example}'])
    utils.create_directory(args_DDPG.data_dir, [f'MultiDDPG_env_{elect_env_example}'])
    # 运行训练
    reward_list, makespan_list, data = runner.run()
    # 绘制移动平均奖励
    avg_rewards = utils.moving_average(reward_list, 100)
    utils.plot_method(range(len(avg_rewards)), avg_rewards,
                      'Moving Average Reward', 'reward',
                      f"{args_DDPG.image_dir}/MultiDDPG_env_{elect_env_example}/avg_reward.png")


if __name__ == "__main__":
    main()