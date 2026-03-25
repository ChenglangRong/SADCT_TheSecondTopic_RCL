import optuna
import torch
import numpy as np
from mainMultiTD3 import MultiMATD3_Runner
from params import args_TD3  # 导入TD3基础参数配置
import random


# 设置随机种子确保实验可复现
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# 定义优化目标函数
def objective(trial):
    # 定义超参数搜索空间（基于您的MATD3实现特点）
    params = {
        # 学习率（对数分布搜索，适合学习率这类小数值）
        "actor_lr": trial.suggest_float("actor_lr", 1e-6, 1e-3, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 1e-5, 3e-3, log=True),

        # 网络结构参数
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512, 1024]),

        # 强化学习核心参数
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),  # 折扣因子
        "tau": trial.suggest_float("tau", 1e-4, 1e-2, log=True),  # 软更新系数

        # 经验回放参数
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 200000, 500000]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),

        # MATD3特有参数
        "policy_delay": trial.suggest_int("policy_delay", 2, 5),  # 策略延迟更新步数
        "noise_std": trial.suggest_float("noise_std", 0.01, 0.2),  # 目标策略噪声标准差
        "noise_clip": trial.suggest_float("noise_clip", 0.5, 2.0),  # 噪声剪辑范围

        # 熵正则化系数（从Agent.py中提取的可优化参数）
        "entropy_coef": trial.suggest_float("entropy_coef", 1e-5, 1e-2, log=True)
    }

    # 更新参数配置到全局参数对象
    args_TD3.actor_lr = params["actor_lr"]
    args_TD3.critic_lr = params["critic_lr"]
    args_TD3.hidden_dim = params["hidden_dim"]
    args_TD3.gamma = params["gamma"]
    args_TD3.tau = params["tau"]
    args_TD3.buffer_size = params["buffer_size"]
    args_TD3.batch_size = params["batch_size"]
    args_TD3.policy_delay = params["policy_delay"]
    args_TD3.noise_std = params["noise_std"]
    args_TD3.noise_clip = params["noise_clip"]

    # 为调参加速，减少训练回合数（根据实际情况调整）
    args_TD3.num_episodes = 30  # 比正式训练少，平衡调参效率和稳定性

    # 选择环境案例（与mainMultiTD3.py保持一致）
    elect_env_example = 6  # 可根据需要修改为7或8
    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        raise ValueError("无效的环境案例选择")

    # 多次运行取平均，减少随机波动影响
    total_rewards = []
    total_makespans = []
    for seed in [42, 43, 44]:  # 3次不同种子的重复实验
        set_seeds(seed)
        # 初始化运行器
        runner = MultiMATD3_Runner(args_TD3, args_env, elect_env_example)
        # 运行训练
        reward_list, makespan_list, _ = runner.run()

        # 记录最后10个回合的平均奖励（稳定期性能）
        if len(reward_list) >= 10:
            total_rewards.append(np.mean(reward_list[-10:]))
        else:
            total_rewards.append(np.mean(reward_list))

        # 记录完工时间（辅助指标，越小越好）
        if makespan_list:
            total_makespans.append(np.mean(makespan_list))

    # 综合优化目标：最大化平均奖励，同时考虑最小化完工时间
    avg_reward = np.mean(total_rewards)
    avg_makespan = np.mean(total_makespans) if total_makespans else 0

    # 加权组合目标（可根据业务需求调整权重）
    combined_score = avg_reward - 0.01 * avg_makespan
    return combined_score


if __name__ == "__main__":
    # 创建优化研究（最大化综合得分）
    study = optuna.create_study(direction="maximize", study_name="MATD3_tuning")

    # 增加可视化回调（需要安装optuna-dashboard：pip install optuna-dashboard）
    try:
        from optuna.integration.wandb import WeightsAndBiasesCallback

        wandb_callback = WeightsAndBiasesCallback(metric_name="combined_score")
        callbacks = [wandb_callback]
    except ImportError:
        callbacks = []  # 若无wandb则不使用

    # 运行参数搜索
    study.optimize(
        objective,
        n_trials=100,  # 尝试100组参数（可根据计算资源调整）
        callbacks=callbacks,
        n_jobs=1  # 单进程运行（多进程需确保环境线程安全）
    )

    # 输出最优结果
    print("\n==================== 最优参数 ====================")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"\n最优综合得分: {study.best_value:.2f}")

    # 保存最优参数到文件
    import json

    with open(f"best_matd3_params_env{6}.json", "w") as f:  # 环境案例6
        json.dump(study.best_params, f, indent=4)
    print("\n最优参数已保存到best_matd3_params.json")