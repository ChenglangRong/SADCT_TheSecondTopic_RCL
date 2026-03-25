import optuna
import torch
import numpy as np
from two_types.MACTD4_2.Agent import MultiAgentMACTD4
from two_types.SADCT_environment2 import Environment


# 定义超参数搜索空间和评估函数
def objective(trial: optuna.Trial):
    # 1. 定义超参数搜索空间
    params = {
        # 学习率（对数分布搜索）
        "actor_lr": trial.suggest_loguniform("actor_lr", 1e-5, 1e-3),
        "critic_lr": trial.suggest_loguniform("critic_lr", 1e-5, 1e-3),
        # 网络结构
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        # 强化学习核心参数
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
        "tau": trial.suggest_uniform("tau", 0.001, 0.1),
        # 经验回放
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 200000]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        # TD3特定参数
        "policy_delay": trial.suggest_int("policy_delay", 2, 5),
        "noise_std": trial.suggest_uniform("noise_std", 0.1, 0.5),
        "noise_clip": trial.suggest_uniform("noise_clip", 0.5, 2.0),
        # CTD4多Critic参数
        "n_critics": trial.suggest_int("n_critics", 2, 4),
        "noise_decay_rate": trial.suggest_uniform("noise_decay_rate", 0.995, 0.9995),
        "min_noise_std": trial.suggest_uniform("min_noise_std", 0.001, 0.05)
    }

    # 2. 环境配置（使用案例6作为基准）
    elect_env_example = 6
    from params import args_6 as args_env
    env = Environment(args_env, args_env.wafer_num)
    state_dim = env.state_dim
    action_dims = env.action_dims
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 初始化多智能体MA-CTD4
    multi_agent = MultiAgentMACTD4(
        state_dim=state_dim,
        hidden_dim=params["hidden_dim"],
        action_dims=action_dims,
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        gamma=params["gamma"],
        tau=params["tau"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        policy_delay=params["policy_delay"],
        noise_std=params["noise_std"],
        noise_clip=params["noise_clip"],
        device=device,
        n_critics=params["n_critics"],
        noise_decay_rate=params["noise_decay_rate"],
        min_noise_std=params["min_noise_std"]
    )

    # 4. 训练并评估（使用较少的episode进行快速评估）
    num_episodes = 50  # 调参阶段可减少episode数量
    total_rewards = []

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            masks = env.get_mask()
            actions = multi_agent.take_actions(state, masks)
            next_state, reward, done = env.step(actions)
            multi_agent.add_experiences(state, actions, reward, next_state, done, masks)

            # 经验池足够大时进行更新
            if all(agent.replay_buffer.size() >= params["batch_size"] for agent in multi_agent.agents):
                multi_agent.update()

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        # 早停机制：如果性能太差可以提前终止
        if i_episode > 10 and np.mean(total_rewards[-10:]) < -1000:
            return -float('inf')

    # 5. 返回最后10个episode的平均奖励作为评估指标
    return np.mean(total_rewards[-10:])


if __name__ == "__main__":
    # 创建Optuna研究并运行优化
    study = optuna.create_study(
        direction="maximize",  # 目标：最大化奖励
        sampler=optuna.samplers.TPESampler(),  # 使用Tree-structured Parzen Estimator采样器
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)  # 早停剪枝器
    )

    # 运行100次试验
    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True,
        n_jobs=1  # 单进程运行（多进程需确保环境线程安全）
    )

    # 输出最优结果
    print("最优超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最优平均奖励: {study.best_value}")

    # 保存最优参数到文件
    import json

    with open("best_CTD4_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)