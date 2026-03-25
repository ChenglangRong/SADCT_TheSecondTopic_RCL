import optuna
import torch
import numpy as np
import random
from mainMultiDDPG import MultiDDPG_Runner
from params import args_DDPG

# 定义参数搜索空间和评估函数
def objective(trial: optuna.Trial):
    # 定义待优化的超参数搜索空间
    params = {
        # 学习率：使用对数均匀分布搜索
        "actor_lr": trial.suggest_loguniform("actor_lr", 1e-5, 1e-3),
        "critic_lr": trial.suggest_loguniform("critic_lr", 1e-5, 1e-3),
        # 折扣因子：0.9到0.999之间
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
        # 软更新系数：较小的范围
        "tau": trial.suggest_loguniform("tau", 1e-4, 1e-2),
        # 隐藏层维度：离散选择
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        # 经验回放批次大小
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        # 熵正则化系数（原代码中的0.001）
        "entropy_coef": trial.suggest_loguniform("entropy_coef", 1e-4, 1e-2)
    }

    # 设置固定参数和试验参数
    args_DDPG.actor_lr = params["actor_lr"]
    args_DDPG.critic_lr = params["critic_lr"]
    args_DDPG.gamma = params["gamma"]
    args_DDPG.tau = params["tau"]
    args_DDPG.hidden_dim = params["hidden_dim"]
    args_DDPG.batch_size = params["batch_size"]

    # 为当前试验设置随机种子确保可复现性
    seed = trial.number  # 使用试验编号作为种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 选择环境案例（保持与原代码一致）
    elect_env_example = 6
    try:
        # 初始化运行器
        runner = MultiDDPG_Runner(args_DDPG, get_env_args(elect_env_example), elect_env_example)

        # 减少训练轮数以加速调参（可根据实际情况调整）
        runner.num_episodes = 100  # 调参阶段用较少轮数

        # 运行训练
        reward_list, makespan_list, _ = runner.run()

        # 计算最后20% episodes的平均奖励作为评估指标
        eval_episodes = int(0.2 * runner.num_episodes)
        if eval_episodes == 0:
            eval_episodes = 1
        mean_reward = np.mean(reward_list[-eval_episodes:])

        # 如果有成功完成的episode，结合makespan进行评估
        if makespan_list:
            mean_makespan = np.mean(makespan_list)
            # 综合评分：奖励越高越好，完工时间越短越好
            score = mean_reward - 0.1 * mean_makespan
            return score
        return mean_reward

    except Exception as e:
        # 捕获异常，避免单个试验失败导致整个调参中断
        print(f"Trial {trial.number} failed: {str(e)}")
        return -float("inf")


# 辅助函数：获取环境参数（根据您的实际参数模块实现）
def get_env_args(example_id):
    from params import args_6
    if example_id == 6:
        return args_6
    else:
        raise ValueError("无效的环境案例ID")


# 执行调参
if __name__ == "__main__":
    # 创建Optuna研究对象，目标是最大化评估分数
    study = optuna.create_study(direction="maximize")

    # 运行优化：设置试验次数（根据计算资源调整）
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 输出最佳参数
    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳评分: {study.best_value}")

    # 保存最佳参数到文件
    import json

    with open("best_DDPG_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)