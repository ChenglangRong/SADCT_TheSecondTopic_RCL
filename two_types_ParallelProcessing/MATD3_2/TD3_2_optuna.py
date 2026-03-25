import optuna
import torch
import numpy as np
import argparse
from mainMultiTD3 import MultiMATD3_Runner  # 正确的类名
from two_types.SADCT_environment2 import Environment  # 导入环境类
from params import args_TD3  # 导入默认参数


# 定义目标函数
def objective(trial: optuna.Trial):
    # 定义待优化的TD3参数搜索空间
    params = {
        "actor_lr": trial.suggest_loguniform("actor_lr", 1e-5, 1e-3),
        "critic_lr": trial.suggest_loguniform("critic_lr", 1e-5, 1e-3),
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
        "tau": trial.suggest_uniform("tau", 0.001, 0.1),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        "policy_delay": trial.suggest_int("policy_delay", 2, 5),
        "noise_std": trial.suggest_uniform("noise_std", 0.1, 0.5),
        "noise_clip": trial.suggest_uniform("noise_clip", 0.5, 2.0),
        "buffer_size": trial.suggest_categorical("buffer_size", [100000, 500000, 1000000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256])
    }

    # 构造MATD3参数命名空间（匹配MultiMATD3_Runner的初始化参数）
    matd3_args = argparse.Namespace(
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        num_episodes=500,  # 调参时减少 episodes 加速搜索
        hidden_dim=params["hidden_dim"],
        gamma=params["gamma"],
        tau=params["tau"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        policy_delay=params["policy_delay"],
        noise_std=params["noise_std"],
        noise_clip=params["noise_clip"],
        ckpt_dir="./tmp_ckpt",
        image_dir="./tmp_images",  # 从默认参数继承必要路径
        data_dir="./tmp_data"
    )

    # 环境参数（使用案例6的参数，可根据需要修改）
    from params import args_6 as args_env

    # 初始化训练器
    runner = MultiMATD3_Runner(matd3_args, args_env, elect_env_example=6)

    # 运行训练并获取性能指标
    try:
        reward_list, makespan_list, _ = runner.run()
        # 以最后10个episode的平均奖励作为优化目标
        avg_reward = np.mean(reward_list[-10:]) if reward_list else -np.inf
        return avg_reward
    except Exception as e:
        print(f"训练出错: {e}")
        return -np.inf


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=1)  # 单进程调试，稳定后可改多进程

    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳平均奖励: {study.best_value}")

    # 保存最佳参数
    import json

    with open("best_TD3_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)