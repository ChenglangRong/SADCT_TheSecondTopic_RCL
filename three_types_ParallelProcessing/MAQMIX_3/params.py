# ========================训练超参数=======================
import argparse

parser = argparse.ArgumentParser()
lr = 1.4904870444041462e-05           # Q网络学习率
mixer_lr = 8e-4                      # 混合网络学习率
num_episodes = 5000                  # 回合数
hidden_dim = 96                      # 隐藏层神经元数量
hidden_dim_mixing = 48               # 混合网络隐藏层
gamma = 0.97404229673434            # 折扣因子
tau = 0.04386239111024942            # 软更新系数
buffer_size = 200000                 # 回放池大小
batch_size = 64                      # 批处理大小
epsilon_start = 1.0                  # ε起始值
epsilon_end = 0.001                  # ε终止值
epsilon_decay = 0.9987916311867949               # ε衰减率
parser_maqmix = argparse.ArgumentParser()
parser_maqmix.add_argument('--lr', type=float, default=lr)  # QMix论文学习率
parser_maqmix.add_argument('--num_episodes', type=int, default=num_episodes)  # 训练回合数
parser_maqmix.add_argument('--hidden_dim', type=int, default=hidden_dim)  # 智能体Q网络隐藏层
parser_maqmix.add_argument('--hidden_dim_mixing', type=int, default=hidden_dim_mixing)  # 混合网络隐藏层
parser_maqmix.add_argument('--gamma', type=float, default=gamma)  # 折扣因子
parser_maqmix.add_argument('--tau', type=float, default=tau)  # 软更新系数
parser_maqmix.add_argument('--buffer_size', type=int, default=buffer_size)  # 回放池大小
parser_maqmix.add_argument('--batch_size', type=int, default=batch_size)  # 批处理大小
parser_maqmix.add_argument('--epsilon_start', type=float, default=epsilon_start)  # ε起始值
parser_maqmix.add_argument('--epsilon_end', type=float, default=epsilon_end)  # ε终止值
parser_maqmix.add_argument('--epsilon_decay', type=float, default=epsilon_decay)  # ε衰减率
parser_maqmix.add_argument('--ckpt_dir', type=str, default='checkpoints/')
parser_maqmix.add_argument('--image_dir', type=str, default='images/')
parser_maqmix.add_argument('--data_dir', type=str, default='data/')
args_MAQMix = parser_maqmix.parse_args()

#===================================================================案例6
robot_actions = ["MT", "WT", "UT", "CT", "LT", "IDLE"]  # 分别表示不持有晶圆移动，等待，卸载，持有晶圆移动，加载
modules_list = ["LL", "PM11", "PM12", "BM1", "PM15", "PM16", "PM21", "PM22", "PM23", "PM24", "PM25", "PM26", "BM2"]
steps_list  =  ["LL", "PM11", "PM12", "BM1", "PM15", "PM16", "PM21", "PM22", "PM23", "PM24", "PM25", "PM26", "BM2"]
process_residency_time_dict = {"LL": [0, 0], "PM11": [85, 20], "PM12": [99, 20], "BM1": [0, 0], "PM15": [94, 20], "PM16": [80, 20],
                               "PM21": [101, 20], "PM22": [120, 20], "PM23": [120, 20], "PM24": [120, 20], "PM25": [120, 20],
                               "PM26": [115, 20], "BM2": [0, 0]}
process_residency_time_list = [[0, 0], [85, 20], [99, 20], [0, 0], [94, 20], [80, 20], [101, 20], [120, 20], [120, 20], [120, 20], [120, 20], [115, 20], [0, 0]]
max_stay_time_dict = {"LL": 0, "PM11": 20, "PM12": 20, "BM1": 0, "PM15": 20, "PM16": 20,
                               "PM21": 20, "PM22": 20, "PM23": 20, "PM24": 20, "PM25": 20,
                               "PM26": 20, "BM2": 0}
max_stay_time_list = [0, 20, 20, 0, 20, 20, 20, 20, 20, 20, 20, 20, 0]
process_time_dict = {"LL": 0, "PM11": 85, "PM12": 99, "BM1": 0, "PM15": 94, "PM16": 80,
                               "PM21": 101, "PM22": 120, "PM23": 120, "PM24": 120, "PM25": 120,
                               "PM26": 115, "BM2": 0}
process_time_list = [0, 85, 99, 0, 94, 80, 101, 120, 120, 120, 120, 115, 0]
wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}
wait_time_list = [0, 0, 0, 0, 0, 0, 38]
unload_time_LL = 10
work_time = 5
move_time = 2

# 动作空间
# robot1负责: LL, PM11, PM12, BM1, BM2, PM15, PM16,
robot1_actions = [
    0,  # 0: 空动作
    [0, 1],  # 1: LL -> PM11
    [1, 2],  # 2: PM11 -> PM12
    [2, 3],  # 3: PM12 -> BM1
    [0, 6],  # 4: LL -> PM16
    [6, 5],  # 5: PM16 -> PM15
    [5, 3],  # 6: PM15 -> BM1
    [4, 0],  # 7: BM2 -> LL
    [0, 3],  # 8: LL -> BM1
]

# robot2负责: PM21, PM22, PM23, PM24, PM25, PM26, BM1, BM2
robot2_actions = [
    0,  # 0: 空动作
    [6, 0],  # 1: BM1 -> PM21
    [0, 1],  # 2: PM21 -> PM22
    [0, 2],  # 3: PM21 -> PM23
    [0, 3],  # 4: PM21 -> PM24
    [0, 4],  # 5: PM21 -> PM25
    [6, 5],  # 6: BM1 -> PM26
    [5, 4],  # 7: PM26 -> PM25
    [5, 3],  # 8: PM26 -> PM24
    [5, 2],  # 9: PM26 -> PM23
    [5, 1],  # 10: PM26 -> PM22
    [6, 1],  # 11: BM1 -> PM22
    [6, 2],  # 12: BM1 -> PM23
    [6, 3],  # 13: BM1 -> PM24
    [6, 4],  # 14: BM1 -> PM25
    [1, 7],  # 15: PM22 -> BM2
    [2, 7],  # 16: PM23 -> BM2
    [3, 7],  # 17: PM24 -> BM2
    [4, 7],  # 18: PM25 -> BM2
    [1, 2],  # 19: PM22 -> PM23
    [1, 3],  # 20: PM22 -> PM24
    [1, 4],  # 21: PM22 -> PM25
    [2, 1],  # 22: PM23 -> PM22
    [2, 3],  # 23: PM23 -> PM24
    [2, 4],  # 24: PM23 -> PM25
    [3, 1],  # 25: PM24 -> PM22
    [3, 2],  # 26: PM24 -> PM23
    [3, 4],  # 27: PM24 -> PM25
    [4, 1],  # 28: PM25 -> PM22
    [4, 2],  # 29: PM25 -> PM23
    [4, 3],  # 30: PM25 -> PM24
]

action_dim = len(robot1_actions) * len(robot2_actions)

wafer_num = 30  # 总晶圆数量
# 两种晶圆各占一半
wafer_type_distribution = [ 1/3, 1/3, 1/3 ]  # 类型1:类型2 = 1:1:1

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