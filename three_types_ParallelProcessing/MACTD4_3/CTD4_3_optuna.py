import optuna
import torch
import numpy as np
import random
from mainMultiCTD4 import MultiMACTD4_Runner
from params import args_CTD4  # 导入基础参数配置


# 确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# 目标函数：返回该组参数的平均奖励
def objective(trial):
    # 设置随机种子确保实验一致性
    set_seed(42)

    # 待优化的超参数搜索空间
    params = {
        # 学习率：对数分布搜索
        "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True),
        # 隐藏层维度
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=64),
        # 折扣因子
        "gamma": trial.suggest_float("gamma", 0.9, 0.999, step=0.001),
        # 软更新系数
        "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
        # 经验回放批次大小
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        # 策略延迟更新步数
        "policy_delay": trial.suggest_int("policy_delay", 1, 5),
        # 噪声参数
        "noise_std": trial.suggest_float("noise_std", 0.1, 1.0),
        "noise_clip": trial.suggest_float("noise_clip", 0.5, 2.0),
        #  Critic数量
        "n_critics": trial.suggest_int("n_critics", 2, 5),
        # 噪声衰减参数
        "noise_decay_rate": trial.suggest_float("noise_decay_rate", 0.99, 0.9995),
        "min_noise_std": trial.suggest_float("min_noise_std", 0.001, 0.1, log=True)
    }

    # 更新参数配置
    args_CTD4.actor_lr = params["actor_lr"]
    args_CTD4.critic_lr = params["critic_lr"]
    args_CTD4.hidden_dim = params["hidden_dim"]
    args_CTD4.gamma = params["gamma"]
    args_CTD4.tau = params["tau"]
    args_CTD4.batch_size = params["batch_size"]
    args_CTD4.policy_delay = params["policy_delay"]
    args_CTD4.noise_std = params["noise_std"]
    args_CTD4.noise_clip = params["noise_clip"]
    args_CTD4.n_critics = params["n_critics"]
    args_CTD4.noise_decay_rate = params["noise_decay_rate"]
    args_CTD4.min_noise_std = params["min_noise_std"]

    # 减少调参时的训练回合数（加速搜索）
    args_CTD4.num_episodes = 100  # 正式训练时可改回更大值

    # 创建运行器并执行训练
    elect_env_example = 6  # 使用案例6进行调参
    from params import args_6 as args_env  # 导入环境参数
    runner = MultiMACTD4_Runner(args_CTD4, args_env, elect_env_example)

    # 运行训练并获取结果
    reward_list, _, _ = runner.run()

    # 使用最后20%的平均奖励作为评价指标（更稳定）
    eval_episodes = int(0.2 * args_CTD4.num_episodes)
    if eval_episodes == 0:
        eval_episodes = 1
    mean_reward = np.mean(reward_list[-eval_episodes:])

    return mean_reward


if __name__ == "__main__":
    # 创建研究并优化目标函数
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    # 运行100次试验（可根据计算资源调整）
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # 输出最佳结果
    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳平均奖励: {study.best_value:.2f}")

    # 保存最佳参数到文件
    import json

    with open("ctd4_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)