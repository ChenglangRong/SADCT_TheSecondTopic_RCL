import optuna
import torch
import numpy as np
from mainMultiDDPG import set_env,MultiAgentDDPG
from params import args_DDPG as base_args_DDPG


# 调参目标函数
def objective(trial):
    # 定义超参数搜索空间
    params = {
        # 学习率：对数分布搜索
        "actor_lr": trial.suggest_loguniform("actor_lr", 1e-5, 1e-3),
        "critic_lr": trial.suggest_loguniform("critic_lr", 1e-5, 1e-3),
        # 隐藏层维度：离散选择
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        # 折扣因子：均匀分布
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
        # 软更新系数：对数分布
        "tau": trial.suggest_loguniform("tau", 1e-4, 1e-2),
        # 经验回放池大小：离散选择
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 200000]),
        # 批处理大小：离散选择
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        # 训练回合数：根据计算资源调整
        "num_episodes": trial.suggest_int("num_episodes", 50, 200)
    }

    # 复制基础参数并更新为当前试验参数
    args = base_args_DDPG
    for key, value in params.items():
        setattr(args, key, value)

    # 创建运行器并执行训练
    elect_env_example = 6  # 选择要优化的环境案例
    runner = set_env(elect_env_example)

    # 覆盖运行器中的参数
    runner.actor_lr = params["actor_lr"]
    runner.critic_lr = params["critic_lr"]
    runner.hidden_dim = params["hidden_dim"]
    runner.gamma = params["gamma"]
    runner.tau = params["tau"]
    runner.buffer_size = params["buffer_size"]
    runner.batch_size = params["batch_size"]
    runner.num_episodes = params["num_episodes"]

    # 重新初始化多智能体（应用新参数）
    runner.multi_agent = runner.multi_agent = MultiAgentDDPG(
        runner.state_dim, runner.hidden_dim, runner.action_dims,
        runner.actor_lr, runner.critic_lr, runner.gamma, runner.tau,
        runner.buffer_size, runner.batch_size, runner.device
    )

    # 运行训练
    reward_list, _, _ = runner.run()

    # 计算最后10%回合的平均奖励作为评估指标
    eval_episodes = max(1, int(params["num_episodes"] * 0.1))
    mean_reward = np.mean(reward_list[-eval_episodes:])

    # 惩罚失败率过高的情况
    fail_rate = np.mean(runner.data['fail'][-eval_episodes:])
    if fail_rate > 0.5:  # 失败率超过50%则扣分
        mean_reward -= 500

    return mean_reward


if __name__ == "__main__":
    # 创建Optuna研究
    study = optuna.create_study(
        direction="maximize",  # 最大化平均奖励
        sampler=optuna.samplers.TPESampler(),  # 使用TPESampler高效搜索
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)  # 早停机制
    )

    # 运行调参（根据计算资源调整n_trials）
    study.optimize(
        objective,
        n_trials=50,  # 试验次数
        n_jobs=1,  # 并行数（根据CPU核心数调整）
        show_progress_bar=True
    )

    # 输出最佳结果
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳平均奖励: {study.best_value}")

    # 保存最佳参数
    import json

    with open("best_ddpg_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)