import optuna
import torch
import random
import numpy as np
from mainMtilQmix import MAQMix_Runner, set_env
from params import args_MAQMix  # 导入基础参数配置


# 定义参数搜索空间和目标函数
def objective(trial: optuna.Trial):
    # 参数搜索空间定义
    params = {
        # 学习率（对数分布搜索）
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        # 折扣因子
        "gamma": trial.suggest_float("gamma", 0.9, 0.999, step=0.001),
        # 软更新系数
        "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
        # 隐藏层维度
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        # 混合网络隐藏层维度
        "hidden_dim_mixing": trial.suggest_categorical("hidden_dim_mixing", [16, 32, 64]),
        # 经验回放池大小
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000]),
        # 批次大小
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        # ε-greedy参数
        "epsilon_start": trial.suggest_float("epsilon_start", 0.9, 1.0),
        "epsilon_end": trial.suggest_float("epsilon_end", 0.01, 0.2),
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.99, 0.999)
    }

    # 覆盖基础参数
    args_MAQMix.lr = params["lr"]
    args_MAQMix.gamma = params["gamma"]
    args_MAQMix.tau = params["tau"]
    args_MAQMix.hidden_dim = params["hidden_dim"]
    args_MAQMix.hidden_dim_mixing = params["hidden_dim_mixing"]
    args_MAQMix.buffer_size = params["buffer_size"]
    args_MAQMix.batch_size = params["batch_size"]
    args_MAQMix.epsilon_start = params["epsilon_start"]
    args_MAQMix.epsilon_end = params["epsilon_end"]
    args_MAQMix.epsilon_decay = params["epsilon_decay"]

    # 为了调参效率，减少训练回合数（可根据实际情况调整）
    args_MAQMix.num_episodes = 200  # 正式训练可改回更大值

    # 固定随机种子确保可复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 创建运行器并开始训练
    elect_env_example = 6  # 使用案例6进行调参
    runner = set_env(elect_env_example)

    # 运行训练
    reward_list, _, _ = runner.run()

    # 以最后100个episode的平均奖励作为评估指标
    if len(reward_list) >= 100:
        avg_reward = np.mean(reward_list[-100:])
    else:
        avg_reward = np.mean(reward_list)

    # 可以添加早停机制避免无效搜索
    if np.isnan(avg_reward) or avg_reward < -1000:
        return -float("inf")

    return avg_reward


if __name__ == "__main__":
    # 创建研究并设置优化方向（最大化平均奖励）
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),  # 使用TPESampler进行高效搜索
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)  # 早停机制
    )

    # 运行参数搜索（设置试验次数）
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 输出最佳结果
    print("最佳参数组合:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳平均奖励: {study.best_value:.2f}")

    # 保存最佳参数到文件
    import json

    with open("best_qmix_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)