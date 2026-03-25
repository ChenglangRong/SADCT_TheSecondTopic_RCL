import optuna
import torch
import random
import numpy as np
from two_types.SADCT_environment2 import Environment
from Agent import MAQMix
import utils


# 定义目标函数：返回需要优化的指标（这里使用平均奖励）
def objective(trial):
    # 定义待优化的超参数搜索空间
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'tau': trial.suggest_float('tau', 0.001, 0.1),
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 128, step=32),
        'hidden_dim_mixing': trial.suggest_int('hidden_dim_mixing', 16, 64, step=16),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.995, 0.999),
        'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000])
    }

    # 固定随机种子以确保实验可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 环境配置（使用案例6）
    from params import args_6 as args_env
    elect_env_example = 6
    env = Environment(args_env, args_env.wafer_num)
    state_dim = env.state_dim
    action_dims = env.action_dims
    n_agents = len(action_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化MAQMix智能体
    maqmix = MAQMix(
        state_dim, params['hidden_dim'], action_dims, n_agents,
        params['lr'], params['gamma'], params['tau'], params['buffer_size'],
        params['batch_size'], device, params['hidden_dim_mixing']
    )

    # 训练配置
    num_episodes = 200  # 调参时可适当减少训练回合数
    epsilon = 1.0
    epsilon_end = 0.05
    reward_list = []

    # 开始训练
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0

        # ε衰减
        epsilon = max(epsilon_end, epsilon * params['epsilon_decay'])

        while not done:
            masks = env.get_mask()
            actions = maqmix.take_actions(state, masks, epsilon)
            next_state, reward, done = env.step(actions)
            masks_tuple = (tuple(masks[0]), tuple(masks[1]))
            maqmix.add_experience(state, actions, reward, next_state, done, masks_tuple)
            state = next_state
            episode_return += reward

            # 经验池足够时进行更新
            if maqmix.replay_buffer.size() >= params['batch_size']:
                maqmix.update()

        reward_list.append(episode_return)

        # 早停机制：如果表现太差可以提前终止
        if i_episode > 50 and np.mean(reward_list[-10:]) < -1000:
            return -np.inf  # 表示该参数组合表现极差

    # 返回最后100个回合的平均奖励（取负是因为Optuna默认最小化目标）
    return -np.mean(reward_list[-min(100, num_episodes):])


def run_hyperparameter_tuning():
    # 创建研究对象并指定优化方向
    study = optuna.create_study(direction='minimize')

    # 运行优化（指定试验次数）
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 输出最佳参数
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳平均奖励: {-study.best_value:.2f}")

    # 保存最佳参数
    import json
    with open('best_qmix_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # 可视化优化过程（需要安装plotly）
    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
    except ImportError:
        print("安装plotly以可视化优化过程: pip install plotly")


if __name__ == "__main__":
    run_hyperparameter_tuning()