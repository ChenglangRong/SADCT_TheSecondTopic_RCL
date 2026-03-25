# ========================训练超参数=======================
import argparse

parser = argparse.ArgumentParser()
# 基础参数（从原Qmix参数继承，适配Qmix-PPO-AC）
# ---------------- Qmix价值侧参数（保留原有最优值） ----------------
lr_q = 1.4904870444041462e-05           # 智能体局部Q网络学习率（原lr）
lr_mixer = 8e-4                         # Qmix混合网络学习率（原mixer_lr）
num_episodes = 5000                     # 总训练回合数
hidden_dim = 96                         # 策略/价值网络隐藏层神经元数量
hidden_dim_mixing = 48                  # Qmix混合网络隐藏层维度
gamma = 0.97404229673434                # 折扣因子
tau = 0.04386239111024942               # 目标网络软更新系数
buffer_size = 200000                    # 经验回放池大小
batch_size = 64                         # 批处理大小
epsilon_start = 1.0                     # ε-greedy起始值
epsilon_end = 0.001                     # ε-greedy终止值
epsilon_decay = 0.9987916311867949      # ε-greedy衰减率

# ---------------- PPO策略侧新增参数（双智能体协作适配） ----------------
lr_actor = 3e-5                         # PPO策略网络学习率
clip_epsilon = 0.2                      # PPO剪切系数（原ppo_clip）
ppo_epochs = 10                         # PPO每批次更新轮数
entropy_coef = 0.01                     # 熵奖励系数（鼓励探索）
update_freq = 5                         # 策略更新频率（每N步更新一次）
# 去掉不必要的GAE参数（Qmix-PPO-AC用Qmix算优势，不需要GAE）

# 统一参数解析器（命名改为QmixPPO，保持和代码调用一致）
parser_QmixPPO = argparse.ArgumentParser(description='Qmix-PPO-AC 双智能体协作算法参数')

# ========== 原有Qmix价值侧参数（完全保留） ==========
parser_QmixPPO.add_argument('--lr_q', type=float, default=lr_q,
                            help='智能体局部Q网络学习率（原Qmix的lr）')
parser_QmixPPO.add_argument('--lr_mixer', type=float, default=lr_mixer,
                            help='Qmix混合网络学习率（原mixer_lr）')
parser_QmixPPO.add_argument('--num_episodes', type=int, default=num_episodes,
                            help='训练总回合数')
parser_QmixPPO.add_argument('--hidden_dim', type=int, default=hidden_dim,
                            help='策略/价值网络隐藏层神经元数量')
parser_QmixPPO.add_argument('--hidden_dim_mixing', type=int, default=hidden_dim_mixing,
                            help='Qmix混合网络隐藏层维度')
parser_QmixPPO.add_argument('--gamma', type=float, default=gamma,
                            help='折扣因子（权衡长期奖励）')
parser_QmixPPO.add_argument('--tau', type=float, default=tau,
                            help='目标网络软更新系数')
parser_QmixPPO.add_argument('--buffer_size', type=int, default=buffer_size,
                            help='经验回放池最大容量')
parser_QmixPPO.add_argument('--batch_size', type=int, default=batch_size,
                            help='批处理大小')
parser_QmixPPO.add_argument('--epsilon_start', type=float, default=epsilon_start,
                            help='ε-greedy探索起始值')
parser_QmixPPO.add_argument('--epsilon_end', type=float, default=epsilon_end,
                            help='ε-greedy探索终止值')
parser_QmixPPO.add_argument('--epsilon_decay', type=float, default=epsilon_decay,
                            help='ε-greedy探索衰减率')

# ========== PPO策略侧新增参数（双智能体适配） ==========
parser_QmixPPO.add_argument('--lr_actor', type=float, default=lr_actor,
                            help='PPO策略网络学习率（双智能体建议3e-5）')
parser_QmixPPO.add_argument('--clip_epsilon', type=float, default=clip_epsilon,
                            help='PPO剪切系数（限制策略更新幅度）')
parser_QmixPPO.add_argument('--ppo_epochs', type=int, default=ppo_epochs,
                            help='PPO每批次更新轮数')
parser_QmixPPO.add_argument('--entropy_coef', type=float, default=entropy_coef,
                            help='熵奖励系数（鼓励双智能体探索）')
parser_QmixPPO.add_argument('--update_freq', type=int, default=update_freq,
                            help='策略网络更新频率（每N步更新一次）')

# ========== 通用保存路径参数（保留原有） ==========
parser_QmixPPO.add_argument('--ckpt_dir', type=str, default='checkpoints/',
                            help='模型保存目录')
parser_QmixPPO.add_argument('--image_dir', type=str, default='images/',
                            help='训练曲线保存目录')
parser_QmixPPO.add_argument('--data_dir', type=str, default='data/',
                            help='训练数据保存目录')

# 解析参数（供主训练文件调用）
args_QmixPPO = parser_QmixPPO.parse_args()

#===================================================================案例6
robot_actions = ["MT", "WT", "UT", "CT", "LT", "IDLE"]  # 分别表示不持有晶圆移动，等待，卸载，持有晶圆移动，加载
modules_list = ["LL", "PM11", "PM12", "BM1", "PM15", "PM16", "PM21", "PM22", "PM23", "PM24", "PM25", "PM26", "BM2"]
steps_list  =  ["LL", "PM11", "PM12", "BM1", "PM15", "PM16", "PM21", "PM22", "PM23", "PM24", "PM25", "PM26", "BM2"]

process_residency_time_dict = {"LL": [0, 0], "PM11": [113, 20], "PM12": [123, 20], "BM1": [0, 0], "PM15": [123, 20], "PM16": [113, 20],
                               "PM21": [105, 20], "PM22": [95, 20], "PM23": [104, 20], "PM24": [100, 20], "PM25": [103, 20],
                               "PM26": [105, 20], "BM2": [0, 0]}

process_residency_time_list = [[0, 0], [113, 20], [123, 20], [0, 0], [123, 20], [113, 20], [105, 20], [95, 20], [104, 20], [100, 20], [103, 20], [105, 20], [0, 0]]

max_stay_time_dict = {"LL": 0, "PM11": 20, "PM12": 20, "BM1": 0, "PM15": 20, "PM16": 20,
                               "PM21": 20, "PM22": 20, "PM23": 20, "PM24": 20, "PM25": 20,
                               "PM26": 20, "BM2": 0}

max_stay_time_list = [0, 20, 20, 0, 20, 20, 20, 20, 20, 20, 20, 20, 0]

process_time_dict = {"LL": 0, "PM11": 113, "PM12": 123, "BM1": 0, "PM15": 123, "PM16": 113,
                               "PM21": 105, "PM22": 95, "PM23": 104, "PM24": 100, "PM25": 103,
                               "PM26": 105, "BM2": 0}

process_time_list = [0, 113, 123, 0, 123, 113, 105, 95, 104, 100, 103, 105, 0]

wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}

wait_time_list = [0, 0, 0, 0, 0, 0, 38]

unload_time_LL = 10
work_time = 5
move_time = 2
# 动作空间
robot1_actions = [
        0,  # 0: 空动作
        [0, 1],  # 1: LL -> PM11
        [0, 6],  # 2: LL -> PM16
        [1, 2],  # 3: PM11 -> PM12
        [1, 5],  # 4: PM11 -> PM15
        [2, 3],  # 5: PM12 -> BM1
        [5, 3],  # 6: PM15 -> BM1
        [6, 5],  # 7: PM16 -> PM15
        [6, 2],  # 8: PM16 -> PM12
        [5, 3],  # 9: PM15 -> BM1
        [4, 3],  # 10: PM14 -> BM1
        [4, 0]  # 11: BM2 -> LL
]

robot2_actions = [
        0,  # 0: 空动作
        [6, 0],  # 1: BM1 -> PM21
        [0, 1],  # 2: PM21 -> PM22
        [1, 2],  # 3: PM22 -> PM23
        [2, 7],  # 4: PM23 -> BM2
        [6, 5],  # 5: BM1 -> PM26
        [5, 4],  # 6: PM26 -> PM25
        [4, 3],  # 7: PM25 -> PM24
        [3, 7]  # 8: PM24 -> BM2
]

action_dim = len(robot1_actions) * len(robot2_actions)

wafer_num = 20  # 总晶圆数量
# 两种晶圆各占一半
wafer_type_distribution = [0.5, 0.5]  # 类型1:类型2 = 1:1

# params for cluster-tool:
parser_6 = argparse.ArgumentParser()
parser_6.add_argument('--robot_actions', default=robot_actions, type=list)
parser_6.add_argument('--modules_list', default=modules_list, type=list)
parser_6.add_argument('--steps_list', default=steps_list, type=list)
parser_6.add_argument('--process_residency_time_dict', default=process_residency_time_dict, type=dict)
parser_6.add_argument('--process_residency_time_list', default=process_residency_time_list, type=list)
parser_6.add_argument('--max_stay_time_dict', default=max_stay_time_dict, type=dict)
parser_6.add_argument('--max_stay_time_list', default=max_stay_time_list, type=list)
parser_6.add_argument('--process_time_dict', default=process_time_dict, type=dict)
parser_6.add_argument('--process_time_list', default=process_time_list, type=list)
parser_6.add_argument('--wait_time_dict', default=wait_time_dict, type=dict)
parser_6.add_argument('--wait_time_list', default=wait_time_list, type=list)
parser_6.add_argument('--unload_time_LL', default=unload_time_LL, type=int)
parser_6.add_argument('--work_time', default=work_time, type=int)
parser_6.add_argument('--move_time', default=move_time, type=int)
parser_6.add_argument('--robot1_actions', default=robot1_actions, type=list)
parser_6.add_argument('--robot2_actions', default=robot2_actions, type=list)
parser_6.add_argument('--action_dim', default=action_dim, type=int)
parser_6.add_argument('--wafer_num', default=wafer_num, type=int)
args_6 = parser_6.parse_args()