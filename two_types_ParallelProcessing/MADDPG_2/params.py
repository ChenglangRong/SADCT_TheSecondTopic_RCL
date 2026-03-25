# ========================训练超参数=======================
import argparse

parser = argparse.ArgumentParser()
actor_lr = 5.049662108271907e-05     # 策略网络学习率
critic_lr = 0.0002639226651009974     # 价值网络学习率
num_episodes = 5000                    # 回合数
hidden_dim = 128                       # 隐藏层神经元数量
gamma = 0.9892089349128511            # 折扣因子
tau = 0.0001685085503622385            # 软更新系数
buffer_size = 10000                  # 回放池大小
batch_size = 64                      # 批处理大小
parser_ddpg = argparse.ArgumentParser()
parser_ddpg.add_argument('--actor_lr', type=float, default=actor_lr)
parser_ddpg.add_argument('--critic_lr', type=float, default=critic_lr)
parser_ddpg.add_argument('--num_episodes', type=int, default=num_episodes)
parser_ddpg.add_argument('--hidden_dim', type=int, default=hidden_dim)
parser_ddpg.add_argument('--gamma', type=float, default=gamma)
parser_ddpg.add_argument('--tau', type=float, default=tau)
parser_ddpg.add_argument('--buffer_size', type=int, default=buffer_size)
parser_ddpg.add_argument('--batch_size', type=int, default=batch_size)
parser_ddpg.add_argument('--ckpt_dir', type=str, default='checkpoints/')
parser_ddpg.add_argument('--image_dir', type=str, default='images/')
parser_ddpg.add_argument('--data_dir', type=str, default='data/')
args_DDPG = parser_ddpg.parse_args()



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