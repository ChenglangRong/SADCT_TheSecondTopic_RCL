"""
单臂双组合设备调度环境
包含PM11,PM12,PM15,PM16，PM21,PM22,PM23,PM24,PM25,PM26，BM1，BM2
robot1负责LL、PM11、PM12、PM15、PM16、BM1、BM2区域
robot2负责PM21-PM26、BM1、BM2区域
三种晶圆类型不同加工路径，机器人动作同步执行
waferA：LL->PM11->PM12->BM1->PM21->PM22->BM2->LL
waferB: LL->PM16->PM15->BM1->PM26->PM25->BM2->LL
waferC: LL->BM1->PM23->PM24->BM2->LL
"""
import numpy as np
import simpy
import argparse

# -------------------------------------------------
#   分析器类
# -------------------------------------------------
class system_profiler(object):
    def __init__(self, modules_name, module_list, loadlock, robots, tot_wafer):
        #self.reward = 0
        self.total_reward = 0
        self.total_wafer = tot_wafer  # 当前需要加工的晶圆数量
        self.system_state = None
        self.state = list()
        self.modules_name = modules_name  # 名称列表
        self.modules_list = module_list  # PM对象列表
        self.entry_wafer = 0
        self.exit_wafer = 0
        self.pre_exit_wafer = 0  # 上一个状态下，完成加工的晶圆数
        self.pre_entry_wafer = 0  # 上一个状态下，待加工的晶圆数
        self.robots = robots   # 机器人列表
        self.loadlock = loadlock
        self.loadlock_idle_time = 0
        self.robot_current_modules = [None, None]  # 两个机器人当前位置
        self.robot_wait_times = [0, 0]
        self.state_values = list()
        self.pre_state = list()
        self.last_actions = [None, None]
        self.current_actions = [None, None]
        self.processing_wafer_count = 0  # 当前系统中需要清空的晶圆数量
        self.wafer_type_count = {1:0, 2:0}  # 记录每种类型晶圆的数量
        self.ll_to_pm_success = False  # LL到PM的成功加载标志
        self.pm_to_pm_success = False  # PM到PM的成功加载标志

        # 初始化状态值
        for name in self.modules_name:
            self.state_values.append({
                'name': name, 'wafer_count': 0, 'process_time_remaining': 0,
                'residency_time_remaining': 0, 'wafer_start_time': 0,
                'wafer_state': None, 'processed_time': 0, 'residency_time': 0,
                'wafer_type': None
            })
            self.pre_state.append({
                'name': name, 'wafer_count': 0, 'process_time_remaining': 0,
                'residency_time_remaining': 0, 'wafer_start_time': 0,
                'wafer_state': None, 'processed_time': 0, 'residency_time': 0,
                'wafer_type': None
            })

    # 更新各PM的状态
    def update_modules_state(self, target, wafer_count, process_time_remaining, residency_time_remaining,
                             pm_state, wafer_start_time, wafer_state, wafer_processed_time,
                             wafer_residency_time, wafer_type=None):
        for item in self.state_values:
            if target == item['name']:
                item['wafer_count'] = wafer_count
                if wafer_count > 0:
                    self.processing_wafer_count += 1
                    if wafer_type:
                        item['wafer_type'] = wafer_type
                item['process_time_remaining'] = process_time_remaining
                item['residency_time_remaining'] = residency_time_remaining
                item['wafer_start_time'] = wafer_start_time
                item['processed_time'] = wafer_processed_time
                item['residency_time'] = wafer_residency_time

    # 更新LL的状态
    def update_loadlock_state(self, entry_count, exit_count, wafer_type=None):
        self.entry_wafer = entry_count  # 待加工的晶圆数量
        self.exit_wafer = exit_count  # 清除回到LL的晶圆数量
        if wafer_type:
            self.wafer_type_count[wafer_type] += 1

    # 更新robot的状态
    def update_robot_state(self, robot_id, robot_current_module, robot_wait_time):
        if robot_current_module == 'LL':
            self.robot_current_modules[robot_id] = 0
        else:
            self.robot_current_modules[robot_id] = self.modules_name.index(robot_current_module) + 1
        self.robot_wait_times[robot_id] = robot_wait_time

    def update_system_wafer_state(self):
        self.processing_wafer_count = 0
        for item in self.state_values:
            if item['wafer_count'] > 0:
                self.processing_wafer_count += 1

    # 获取状态
    def get_state(self):
        self.state = list()
        self.state.append(self.exit_wafer)              # 加工完成后回到LL的晶圆数量
        self.state.append(self.entry_wafer)              # 待加工的晶圆数量

        # 两个机器人的当前位置
        self.state.append(self.robot_current_modules[0])
        self.state.append(self.robot_current_modules[1])
        self.state.append(self.processing_wafer_count)  # 系统中当前正在加工的晶圆数量

        # 添加两种晶圆的计数
        self.state.append(self.wafer_type_count[1])
        self.state.append(self.wafer_type_count[2])

        for item in self.state_values:
            self.state.append(item['wafer_count'])
            self.state.append(0 if item['wafer_count'] == 0 else item['wafer_type'])
            self.state.append(item['process_time_remaining'])
            self.state.append(item['residency_time_remaining'])
            self.state.append(item['processed_time'])
            self.state.append(item['residency_time'])

        return self.state

    def get_state_dim(self):
        return len(self.get_state())

    # 获取奖励
    def get_reward(self, robot_action_flags, fail_flags, bottleneck_time):
        success_flag = False
        #self.reward = 0
        current_increment = 0

        # 失败惩罚
        if any(fail_flags):
            current_increment += -5000
            success_flag = False

        # 完成奖励
        if self.exit_wafer == self.total_wafer:
            current_increment += 2000
            success_flag = True

        # 晶圆完成奖励
        if self.robots[0].current_module == 'LL' and self.pre_exit_wafer + 1 == self.exit_wafer:
            current_increment += 50
            self.pre_exit_wafer = self.exit_wafer

        # 加工中晶圆奖励
        #current_increment += self.processing_wafer_count * 10

        # 动作时间惩罚
        #for robot in self.robots:
        #    if robot.take_action_time > 0:
        #        self.reward -= robot.take_action_time * 0.6

        # 探索奖励
        for flag in robot_action_flags:
            if flag:
                current_increment += 10

        # LL中的晶圆被取出并正确加载到PM的奖励
        if self.ll_to_pm_success:
            current_increment += 100
            self.ll_to_pm_success = False  # 重置标志

        # PM间成功转移晶圆的奖励
        if self.pm_to_pm_success:
            current_increment += 100
            self.pm_to_pm_success = False  # 重置标志

        self.total_reward += current_increment

        return self.total_reward, success_flag

    # 打印信息
    def print_info(self, reward, env):
        print('------------------------------------------------执行动作完毕后的系统状态--------------')
        print('LL中尚未加工的晶圆数:   {0}'.format(self.entry_wafer))
        print('加工完成后返回LL的晶圆数:   {0}'.format(self.exit_wafer))
        print('当前系统中正在加工的晶圆数量:   {0}'.format(self.processing_wafer_count))
        print(f"robot1当前位于:{self.robots[0].current_module}, 等待时间: {self.robots[0].wait_time}")
        print(f"robot2当前位于:{self.robots[1].current_module}, 等待时间: {self.robots[1].wait_time}")
        print(f"robot1完成动作时间: {self.robots[0].take_action_time}")
        print(f"robot2完成动作时间: {self.robots[1].take_action_time}")

        for item in self.state_values:
            print('{0}\t是否存在晶圆:{1}\t晶圆类型:{2}\t晶圆剩余加工时间:{3}\t剩余驻留时间:{4}\t晶圆已经加工时间:{5}\t晶圆已经驻留时间时间:{6}\t'
                  '晶圆在当前PM开始加工时间:{7}'.format(
                    item['name'], item['wafer_count'],
                    item['wafer_type'] if item['wafer_count'] > 0 else 'N/A',
                    item['process_time_remaining'], item['residency_time_remaining'],
                    item['processed_time'], item['residency_time'],
                    item['wafer_start_time']))

        print('当前奖励: {0}  当前时间:{1}'.format(reward, env.now))


# -------------------------------------------------
#   晶圆类
# -------------------------------------------------
class Wafer(object):
    def __init__(self, id, state, is_virtual, process_time, max_stay_time, wait_time, wafer_type):
        self.id = id
        self.state = state  # 晶圆当前所在的加工模块
        self.is_broken = False  # 晶圆是否损坏
        self.is_virtual = is_virtual  # 晶圆是否为虚拟晶圆
        self.process_time = process_time  # 加工时间列表
        self.max_stay_time = max_stay_time  # 加工完成后允许在PM驻留的最大时间，列表
        self.wait_time = wait_time  # 等待时间
        self.wafer_type = wafer_type  # 晶圆类型(1或2)
        self.process_step = 0  # 当前处理步骤
        self.path = self.get_path()  # 晶圆的加工路径
        self.prev_state = None  # 记录上一个所在模块

    def get_path(self):
        """根据晶圆类型返回对应的加工路径"""
        if self.wafer_type == 1:  # waferA路径
            return ["LL", "PM11", "PM12", "BM1", "PM21",
                    ["PM22", "PM23", "PM24", "PM25"],
                    "BM2", "LL"    ]

        elif self.wafer_type == 2:  # waferB路径
            return ["LL", "PM16", "PM15", "BM1", "PM26",
                    ["PM22", "PM23", "PM24", "PM25"],
                    "BM2", "LL"    ]

        elif self.wafer_type == 3:  # waferC路径
            return [  "LL", "BM1",
                    ["PM22", "PM23", "PM24", "PM25"],
                    ["PM22", "PM23", "PM24", "PM25"],
                    "BM2", "LL"    ]
        else:
            return []

    def get_next_step_info(self):
        """获取当前步骤的下一个步骤信息（包含是否为并行模块）"""
        if self.process_step >= len(self.path) - 1:
            return None, True  # (模块/模块列表, 是否为最后一步)

        next_step_index = self.process_step + 1
        next_step = self.path[next_step_index]
        return next_step, False


    def is_valid_next_module(self, module):
        """检查模块是否为当前步骤的有效下一个模块（支持并行模块）"""
        next_step, is_final = self.get_next_step_info()
        if is_final or next_step is None:
            return False

        # 并行模块：只要是列表中的一个就有效
        if isinstance(next_step, list):
            return module in next_step
        # 非并行模块：必须完全匹配
        else:
            return module == next_step


# -------------------------------------------------
#   PM类（包含BM特殊模块）
# -------------------------------------------------
class ProcessModule(object):
    def __init__(self, env, name, process_time, max_stay_time, wait_time, pre_module, next_module,
                 loadlock, robot, module_list, is_bm=False, wafer_types_allowed=None, profiler=None):
        self.env = env
        self.name = name
        self.process_time = process_time
        self.max_stay_time = max_stay_time
        self.wait_time = wait_time
        self.pre_module = pre_module
        self.next_module = next_module
        self.loadlock = loadlock
        self.robot = robot
        self.module_list = module_list
        self.state = None
        self.store = simpy.Store(self.env)
        self.monitoring_data = []
        self.wafer_start_time = 0  # 当前晶圆进入PM的时间
        self.last_wafer_left_time = 0  # 上一个晶圆，离开的时间
        self.idle_time = 0
        self.fail = False  # 失败标记
        self.is_bm = is_bm  # 是否为特殊模块BM
        self.wafer_types_allowed = wafer_types_allowed or []  # 允许处理的晶圆类型
        self.profiler = profiler  # 保存分析器引用

    # PM加工进程（BM模块无加工时间）
    def processing(self, process_time):
        if not self.is_bm:  # BM模块不进行加工
            yield self.env.timeout(process_time)
        self.store.items[0].state = self.name  # 加工完成，标记晶圆状态为当前模块名称
        self.store.items[0].process_step += 1  # 更新晶圆处理步骤

    # 加载晶圆到PM
    def load(self, wafer):
        if self.is_bm and self.store.items.__len__() == 1:
            print(f"{self.name}（Buffer）加载晶圆失败：当前已有1个晶圆，不允许重复加载")
            self.fail = True
            return

        if self.store.items.__len__() == 1:
            print(f"{self.name}加载晶圆失败,已有晶圆存在")
            self.fail = True
            return
        # 检查晶圆类型是否允许在此PM处理
        if self.wafer_types_allowed and wafer.wafer_type not in self.wafer_types_allowed:
            print(f"{self.name}不允许处理类型{wafer.wafer_type}的晶圆")
            self.fail = True
            return
        # 关键修改：检查当前模块是否为晶圆的有效下一个模块（支持并行模块）
        if not wafer.is_valid_next_module(self.name):
            next_step, _ = wafer.get_next_step_info()
            print(f"{self.name}加载晶圆失败，加工顺序不匹配。预期{next_step}，实际{self.name}")
            self.fail = True
            return

        self.fail = False
        self.max_stay_time = wafer.max_stay_time[self.name] if self.name in wafer.max_stay_time else 0
        self.process_time = wafer.process_time[self.name] if self.name in wafer.process_time else 0
        self.store.put(wafer)
        self.wafer_start_time = self.env.now  # 记录晶圆开始进入PM的时间

        # 判断晶圆来源并设置奖励标志
        if wafer.prev_state == "LL":
            self.profiler.ll_to_pm_success = True  # 从LL加载到PM
        elif wafer.prev_state in self.module_list:
            self.profiler.pm_to_pm_success = True  # 从PM加载到另一个PM

        self.env.process(self.processing(self.process_time))  # 启动加工进程

    # 从PM卸载晶圆（BM模块无驻留时间约束）
    def unload(self):
        # 记录初始检查时间
        initial_check_time = self.env.now

        # 等待最多30秒，检查是否有晶圆
        while self.env.now - initial_check_time < 31:
            if self.store.items:  # 检查是否有晶圆
                break
            yield self.env.timeout(1)  # 每1秒检查一次

        # 30秒后仍无晶圆则失败
        if not self.store.items:
            self.fail = True
            print(f"{self.name}卸载晶圆失败，等待30秒后仍不存在晶圆")
            return None

        # BM模块不检查加工和驻留时间
        if not self.is_bm:
            if self.env.now - self.wafer_start_time < self.process_time:
                self.fail = True
                print(f"{self.name}卸载晶圆失败，未到达加工时间")
                return None

            if self.env.now - self.wafer_start_time > self.process_time + self.max_stay_time:
                self.fail = True
                print(f"{self.name}卸载晶圆失败，违反了驻留时间约束")
                wafer = yield self.store.get()  # 正确获取晶圆
                wafer.is_broken = True
                return wafer

        self.fail = False
        wafer = yield self.store.get()  # 正确获取晶圆（关键修复）
        self.last_wafer_left_time = self.env.now
        self.wafer_start_time = 0
        return wafer

    # 获取当前PM的晶圆数量
    def get_wafer_count(self):
        return self.store.items.__len__()

    # 获取当前PM的晶圆
    def get_current_wafer(self):
        if self.store.items.__len__() == 1:
            return self.store.items[-1]
        return None

    # 获取当前PM的晶圆的状态
    def get_wafer_state(self):
        if self.store.items.__len__() == 1:
            return self.store.items[-1].is_broken
        return None

    # 获取当前PM的晶圆剩余的加工时间
    def get_process_remaining_time(self):
        if self.is_bm:  # BM模块无加工时间
            return 0
        if self.store.items.__len__() != 0:  # 当前存在晶圆
            wafer = self.store.items[0]
            self.process_time = wafer.process_time[self.name] if self.name in wafer.process_time else 0
            return max(0, self.process_time - (self.env.now - self.wafer_start_time))
        return 0

    # 获取当前PM的晶圆剩余的驻留时间（BM模块无约束）
    def get_residency_remaining_time(self):
        if self.is_bm:  # BM模块无驻留时间约束
            return float(0)
        if self.store.items.__len__() != 0:  # 当前存在晶圆
            wafer = self.store.items[0]
            self.max_stay_time = wafer.max_stay_time[self.name] if self.name in wafer.max_stay_time else 0
            self.process_time = wafer.process_time[self.name] if self.name in wafer.process_time else 0
            upper_limit = self.max_stay_time + self.process_time
            residence_time = self.env.now - self.wafer_start_time
            # 当驻留时间超过上限时，不仅返回0，还直接标记失败
            remaining = max(0, upper_limit - residence_time)
            if remaining == 0 and residence_time > upper_limit:
                self.fail = True  # 标记PM失败
            return remaining
        return 0

    # 获取当前PM的晶圆已经加工时间
    def get_wafer_processed_time(self):
        if self.store.items.__len__() == 0:  # 当前不存在晶圆
            return -1
        if self.is_bm:  # BM模块无加工时间
            return 0
        processed = self.env.now - self.wafer_start_time
        return min(processed, self.process_time)

    def get_wafer_residency_time(self):
        if self.store.items.__len__() == 0:  # 当前不存在晶圆
            return -1
        return self.env.now - self.wafer_start_time


# -------------------------------------------------
#   LL类
# -------------------------------------------------
class Loadlock(object):
    def __init__(self, env, name, wafer_num, module_list):
        self.env = env
        self.name = name
        self.wafer_num = wafer_num  # 系统中的晶圆数量
        self.entry_stores = {1: simpy.Store(self.env), 2: simpy.Store(self.env),
                             3:simpy.Store(self.env)}  # 按类型存储待加工晶圆
        self.exit_store = simpy.Store(self.env)  # 存储已完成晶圆
        self.monitoring_data = []
        self.system_state = "initial_transient"
        self.fail = False
        self.virtual_wafer_num = 0
        self.entry_system_wafer_num = 0
        self.module_list = module_list
        self.last_unload_time = 0
        self.idle_time = 0
        self.system_wafer_count = 0
        self.wafer_start_time = 0
        self.wafer_type_count = {1: 0, 2:0, 3:0}  # 记录每种类型晶圆的数量
        self.fail_reason = ""  # 记录失败原因
        # 添加交替生产控制变量
        self.next_wafer_type = 1  # 初始先生产类型1
        self.wafer_sequence_count = 0  # 用于跟踪交替序列

    # 从LL卸载晶圆 - 修改为交替生产逻辑
    def unload(self, wafer_type=None):
        # 忽略传入的wafer_type，使用内部交替逻辑
        target_type = self.next_wafer_type
        # 检查目标类型是否有可用晶圆
        if self.entry_stores[target_type].items.__len__() == 0:
            # 如果目标类型已耗尽，尝试切换到另一种类型
            other_type = [t for t in [1, 2, 3] if t != target_type]
            found = False
            for t in other_type:
                if self.entry_stores[t].items.__len__() > 0:
                    target_type = t
                    found = True
                    break
            if not found:
                self.fail = True
                self.fail_reason = "所有类型的待加工晶圆已耗尽"
                print(f"卸载失败：所有类型的待加工晶圆已耗尽")
                return None

        # 执行卸载操作
        self.fail = False
        self.fail_reason = ""
        wafer = self.entry_stores[target_type].get()
        self.last_unload_time = self.env.now + 10
        self.system_wafer_count += 1
        self.wafer_type_count[target_type] -= 1  # 更新类型计数

        # 设置晶圆原先位置为LL
        if type(wafer) is simpy.resources.store.StoreGet:
            wafer.value.prev_state = "LL"
        else:
            wafer.prev_state = "LL"

        # 切换下一个要生产的类型（循环1→2→3→1...）
        self.next_wafer_type = 1 if target_type == 3 else target_type + 1

        if type(wafer) is simpy.resources.store.StoreGet:
            return wafer.value
        return wafer

    def initialize(self, wafers):
        # 按类型放入对应队列
        for wafer in wafers:
            self.entry_stores[wafer.wafer_type].put(wafer)
            self.wafer_type_count[wafer.wafer_type] += 1
        print(
            f"LL初始化完成: 类型1晶圆={self.wafer_type_count[1]}, 类型2晶圆={self.wafer_type_count[2]}, 类型3晶圆={self.wafer_type_count[3]}")

    # 加载晶圆到LL
    def load(self, wafer):
        # 检查晶圆是否完成所有必要步骤
        if wafer.wafer_type == 1 and wafer.process_step == 6:  # waferA需要完成6个步骤
            pass
        elif wafer.wafer_type == 2 and wafer.process_step == 6:  # waferB需要完成6个步骤
            pass
        elif wafer.wafer_type == 3 and wafer.process_step == 4:  # waferC需要完成4个步骤
            pass
        else:
            self.fail = True
            print(f"W{wafer.id}加载到LL失败，晶圆尚未完成！")
            return

        self.fail = False
        self.exit_store.put(wafer)
        self.wafer_type_count[wafer.wafer_type] -= 1
        if self.exit_store.items.__len__() == self.wafer_num:  # 已经清空完毕
            print(f"\n==========================加工完毕:{self.env.now}===========================")
            print(f"清空用时:{self.env.now}")


    # 剩余加工晶圆数量（所有类型总和）
    def get_remaining_wafer_count(self):
        return sum(store.items.__len__() for store in self.entry_stores.values())

    # 已完成加工晶圆数量
    def get_finished_wafer_count(self):
        return self.exit_store.items.__len__()


# -------------------------------------------------
#   Robot类（支持两个机器人）
# -------------------------------------------------
class Robot(object):
    def __init__(self, env, name, move_time, work_time, unload_time_LL, current_module, loadlock, robot_id):
        self.env = env
        self.name = name
        self.store = simpy.Store(self.env)
        self.unload_time_LL = unload_time_LL
        self.move_time = move_time
        self.work_time = work_time
        self.monitoring_data = []
        self.current_module = current_module
        self.pre_module = None
        self.wafer_start_time = 0
        self.wait_time = 0
        self.loadlock = loadlock
        self.take_action_time = 0
        self.action_start_time = 0
        self.fail = False
        self.carrying_wafer_type = None  # 记录当前携带的晶圆类型
        self.robot_id = robot_id  # 机器人ID: 0表示robot1, 1表示robot2

    # 加载晶圆到robot
    def load(self, wafer):
        if self.store.items.__len__() == 1:
            self.fail = True
            return
        self.fail = False
        if self.current_module == "LL":
            yield self.env.timeout(self.unload_time_LL)
        else:
            yield self.env.timeout(self.work_time)
        self.store.put(wafer)
        self.wafer_start_time = self.env.now
        self.carrying_wafer_type = wafer.wafer_type  # 记录晶圆类型

    # 从robot卸载晶圆
    def unload(self):
        if self.store.items.__len__() == 0:
            self.fail = True
            return None
        self.fail = False
        wafer = self.store.get()
        self.wafer_start_time = 0
        self.carrying_wafer_type = None  # 清空携带的晶圆类型
        yield self.env.timeout(self.work_time)
        if type(wafer) is simpy.resources.store.StoreGet:
            return wafer.value
        return wafer

    # 获取当前robot的晶圆数量
    def get_wafer_count(self):
        return self.store.items.__len__()

    # robot移动到目标模块
    def move(self, target):
        if self.current_module != target:
            yield self.env.timeout(self.move_time)
            self.pre_module = self.current_module
            self.current_module = target


# -------------------------------------------------
#   环境类
# -------------------------------------------------
class Environment(object):

    def __init__(self, args, wafer_num, wafer_type_distribution=None):
        self.env = simpy.Environment()
        self.specific_pms_count1 = 0
        self.specific_pms_count2 = 0
        self.specific_pms_count3 = 0
        self.specific_pms_count4 = 0
        self.specific_pms_count5 = 0
        self.specific_pms_count6 = 0
        self.specific_pms_count7 = 0
        self.specific_pms_count8 = 0
        self.specific_pms_count9 = 0
        self.specific_pms_count10 = 0
        self.specific_pms_count11 = 0
        self.specific_pms_count12 = 0
        self.specific_pms_count13 = 0

        # robot1负责: LL, PM11, PM12, BM1, BM2, PM15, PM16,
        self.robot1_actions = [
            0,                  # 0: 空动作
            [0, 1],             # 1: LL -> PM11
            [1, 2],             # 2: PM11 -> PM12
            [2, 3],             # 3: PM12 -> BM1
            [0, 6],             # 4: LL -> PM16
            [6, 5],             # 5: PM16 -> PM15
            [5, 3],             # 6: PM15 -> BM1
            [4, 0],             # 7: BM2 -> LL
            [0, 3],             # 8: LL -> BM1
        ]

        # robot2负责: PM21, PM22, PM23, PM24, PM25, PM26, BM1, BM2
        self.robot2_actions = [
            0,                  # 0: 空动作
            [6, 0],             # 1: BM1 -> PM21
            [0, 1],             # 2: PM21 -> PM22
            [0, 2],             # 3: PM21 -> PM23
            [0, 3],             # 4: PM21 -> PM24
            [0, 4],             # 5: PM21 -> PM25
            [6, 5],             # 6: BM1 -> PM26
            [5, 4],             # 7: PM26 -> PM25
            [5, 3],             # 8: PM26 -> PM24
            [5, 2],             # 9: PM26 -> PM23
            [5, 1],             # 10: PM26 -> PM22
            [6, 1],             # 11: BM1 -> PM22
            [6, 2],             # 12: BM1 -> PM23
            [6, 3],             # 13: BM1 -> PM24
            [6, 4],             # 14: BM1 -> PM25
            [1, 7],             # 15: PM22 -> BM2
            [2, 7],             # 16: PM23 -> BM2
            [3, 7],             # 17: PM24 -> BM2
            [4, 7],             # 18: PM25 -> BM2
            [1, 2],             # 19: PM22 -> PM23
            [1, 3],             # 20: PM22 -> PM24
            [1, 4],             # 21: PM22 -> PM25
            [2, 1],             # 22: PM23 -> PM22
            [2, 3],             # 23: PM23 -> PM24
            [2, 4],             # 24: PM23 -> PM25
            [3, 1],             # 25: PM24 -> PM22
            [3, 2],             # 26: PM24 -> PM23
            [3, 4],             # 27: PM24 -> PM25
            [4, 1],             # 28: PM25 -> PM22
            [4, 2],             # 29: PM25 -> PM23
            [4, 3],             # 30: PM25 -> PM24
        ]

        # 模块列表（更新为指定的加工模块）
        self.modules_list = ["LL", "PM11", "PM12", "BM1", "PM15", "PM16",
                            "PM21", "PM22", "PM23", "PM24", "PM25", "PM26", "BM2"]
        self.steps_list = self.modules_list.copy()

        # 其他参数初始化
        self.process_residency_time_dict = args.process_residency_time_dict if args else {}
        self.process_residency_time_list = args.process_residency_time_list if args else []
        self.max_stay_time_dict = args.max_stay_time_dict if args else {}
        self.max_stay_time_list = args.max_stay_time_list if args else []
        self.process_time_dict = args.process_time_dict if args else {}
        self.process_time_list = args.process_time_list if args else []
        self.wait_time_dict = args.wait_time_dict if args else {}
        self.wait_time_list = args.wait_time_list if args else []
        self.unload_time_LL = args.unload_time_LL if args else 10
        self.work_time = args.work_time if args else 5
        self.move_time = args.move_time if args else 2
        self.wafer_num = wafer_num
        self.wafer_type_distribution = wafer_type_distribution or [ 1/3, 1/3, 1/3 ]  # 默认两种晶圆各占50%

        self.robots = []  # 存储两个机器人
        self.profiler = None  # 先声明profiler属性，避免未定义
        self.loadlock = None
        self.modules = list()
        self.bottleneck_time = 0
        self.state = list()
        self.reward = 0
        self.done = False
        self.fail_flags = [False, False]  # 两个机器人的失败标记
        self.success_flag = False
        self.state_dim = 0
        self.action_dims = [len(self.robot1_actions), len(self.robot2_actions)]  # 两个机器人的动作维度
        self.robot_current_modules = ['LL', 'PM21']  # 初始位置
        self.initialize()

    def initialize(self):
        self.env = simpy.Environment()  # 创建环境

        # 生成晶圆
        wafers = self.generate_wafers(self.wafer_num, self.wafer_type_distribution)

        # 生成loadlock对象
        self.loadlock = Loadlock(self.env, "LL", self.wafer_num, self.modules_list)
        self.loadlock.initialize(wafers)

        # 定义各PM允许处理的晶圆类型
        pm_wafer_types = {
            # robot1负责的模块
            "PM11": [1], "PM12": [1], "PM15": [2], "PM16": [2],
            # robot2负责的模块
            "PM21": [1],
            "PM22": [1,2,3], "PM23": [1,2,3],
            "PM24": [1,2,3], "PM25": [1,2,3],
            "PM26": [2],
            # BM模块允许两种类型
            "BM1": [1, 2, 3], "BM2": [1, 2, 3]
        }

        # 生成两个机器人对象
        self.robots = [
            Robot(self.env, "robot1", self.move_time, self.work_time, self.unload_time_LL,
                  self.robot_current_modules[0], self.loadlock, 0),
            Robot(self.env, "robot2", self.move_time, self.work_time, self.unload_time_LL,
                  self.robot_current_modules[1], self.loadlock, 1)
        ]

        # 初始化所有PM（包含BM模块）
        self.modules.clear()
        for name in self.modules_list[1:]:  # 跳过LL
            # 判断是否为BM模块
            is_bm = name in ["BM1", "BM2"]
            # 确定前后模块
            idx = self.modules_list.index(name)
            pre_module = self.modules_list[idx-1] if idx > 0 else None
            next_module = self.modules_list[idx+1] if idx < len(self.modules_list)-1 else None

            # 确定负责的机器人：robot1负责LL、PM11、PM12、PM15、PM16、BM1、BM2
            # robot2负责PM21-PM26、BM1、BM2
            if name in ["PM11", "PM12", "PM15", "PM16"]:
                robot = self.robots[0]
            elif name in ["PM21", "PM22", "PM23", "PM24", "PM25", "PM26"]:
                robot = self.robots[1]
            else:  # BM1和BM2由两个机器人共享
                robot = self.robots[0]  # 默认归属，实际可由两个机器人操作

            # 创建PM对象
            PM = ProcessModule(
                self.env, name,
                self.process_time_dict.get(name, 0),
                self.max_stay_time_dict.get(name, 0),
                self.wait_time_dict.get(name, 0),
                pre_module, next_module,
                self.loadlock,
                robot,
                self.modules,
                is_bm=is_bm,
                wafer_types_allowed=pm_wafer_types.get(name, []),
                profiler = self.profiler  # 传入分析器引用
            )
            self.modules.append(PM)

        # 初始化事件（支持双机器人同步）
        self.event_entry = self.env.event()
        self.event_exit = self.env.event()
        self.event_hdlr = self.env.event()
        self.event_steps = [self.env.event(), self.env.event()]  # 两个机器人的步骤事件
        self.event_action = self.env.event()
        self.robot1_events = {event: self.env.event() for event in ["MT", "WT", "UT", "CT", "LT", "IDLE"]}
        self.robot2_events = {event: self.env.event() for event in ["MT", "WT", "UT", "CT", "LT", "IDLE"]}

        # 初始时机器人处于IDLE状态
        if not self.robot1_events["IDLE"].triggered:
            self.robot1_events["IDLE"].succeed()
        if not self.robot2_events["IDLE"].triggered:
            self.robot2_events["IDLE"].succeed()

        # 初始化动作、奖励、状态和各种标记
        self.actions = [0, 0]
        self.reward = 0
        self.fail_flags = [False, False]
        self.success_flag = False
        self.done = False
        self.robot_action_flags = [False, False]
        self.state = []
        self.wafer_in_proc = 0

        self.curr_nope_count = 0
        self.profiler = self.init_system_profiler()
        self.state_dim = self.profiler.get_state_dim()
        self.bottleneck_time = sum(self.process_time_list)
        self.process_handler = self.env.process(self.proc_handler())  # 处理器进程

    # 生成晶圆
    def generate_wafers(self, num, distribution):
        wafers = []
        # 确定两种晶圆的数量
        type1_num = int(num * distribution[0])
        type2_num = int(num * distribution[1])
        type3_num = int(num * distribution[2])

        # 生成类型1晶圆 (waferA)
        for i in range(type1_num):
            process_time = {
                "PM11": 85, "PM12": 99, "BM1": 0,  # BM无加工时间
                "PM21": 101, "PM22": 120, "PM23": 120, "PM24": 120, "PM25": 120,
                "BM2": 0
            }
            max_stay_time = {
                "PM11": 20, "PM12": 20, "BM1": 0,  # BM无驻留时间约束
                "PM21": 20, "PM22": 20, "PM23": 20, "PM24": 20, "PM25": 20,
                "BM2": 0
            }
            wafer = Wafer(i+1, "LL", False, process_time, max_stay_time, 0, 1)
            wafers.append(wafer)

        # 生成类型2晶圆 (waferB)
        for i in range(type2_num):
            process_time = {
                "PM16": 80, "PM15": 94, "BM1": 0,  # BM无加工时间
                "PM26": 115, "PM22": 120, "PM23": 120, "PM24": 120, "PM25": 120,
                "BM2": 0
            }
            max_stay_time = {
                "PM16": 20, "PM15": 20, "BM1": 0,  # BM无驻留时间约束
                "PM26": 20, "PM22": 20, "PM23": 20, "PM24": 20, "PM25": 20,
                "BM2": 0
            }
            wafer = Wafer(type1_num + i + 1, "LL", False, process_time, max_stay_time, 0, 2)
            wafers.append(wafer)

        # 生成类型3晶圆 (waferC)
        for i in range(type3_num):
            process_time = {
                "BM1": 0,  # BM无加工时间
                "PM22": 120, "PM23": 120, "PM24": 120, "PM25": 120,
                "BM2": 0
            }
            max_stay_time = {
                "BM1": 0,  # BM无驻留时间约束
                "PM22": 20, "PM23": 20, "PM24": 20, "PM25": 20,
                "BM2": 0
            }
            wafer = Wafer(type1_num + type2_num + i + 1, "LL", False, process_time, max_stay_time, 0, 3)
            wafers.append(wafer)

        return wafers

    # 环境的外部接口：重置环境
    def reset(self):
        del self.env
        self.initialize()
        # 更新状态
        for pm in self.modules:
            wafer = pm.get_current_wafer()
            wafer_type = wafer.wafer_type if wafer else None
            self.profiler.update_modules_state(
                pm.name, pm.store.items.__len__(),
                pm.get_process_remaining_time(),
                pm.get_residency_remaining_time(),
                pm.state, pm.wafer_start_time,
                pm.get_wafer_state(),
                pm.get_wafer_processed_time(),
                pm.get_wafer_residency_time(),
                wafer_type
            )

        self.profiler.update_loadlock_state(
            self.loadlock.get_remaining_wafer_count(),
            self.loadlock.get_finished_wafer_count()
        )
        self.profiler.update_robot_state(0, self.robots[0].current_module, self.robots[0].wait_time)
        self.profiler.update_robot_state(1, self.robots[1].current_module, self.robots[1].wait_time)
        self.profiler.update_system_wafer_state()
        self.state = self.profiler.get_state()
        return self.state

    # 环境的外部接口：执行一步动作（双机器人同步）
    def step(self, actions):
        self.fail_flags = [False, False]
        self.actions = actions  # actions是一个列表，包含两个机器人的动作
        self.event_action.succeed()  # 触发事件，通知处理器处理动作
        print("******开始时间：", self.env.now)
        for i in range(2):
            self.robots[i].action_start_time = self.env.now
        self.env.run(self.event_hdlr)  # 运行环境直到两个机器人都完成动作
        self.event_hdlr = self.env.event()  # 重置事件

        print("******结束时间：", self.env.now)
        self.f_time = self.env.now
        obs, reward, done = self.get_observation()
        return obs, reward, done

    def get_mask(self):
        """生成两个机器人的动作掩码，1表示动作有效，0表示动作无效
        逻辑：初始全允许（全1），匹配特定模块状态时，仅保留有效动作，其余设为非法（0）
        """
        has_wafer_pms = sorted([pm.name for pm in self.modules if len(pm.store.items) > 0])
        pms_key = tuple(has_wafer_pms)
        mask_mapping = {
            (): (
                [1 if i == 1 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11",): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM16"): (
                [1 if i == 8 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM16"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 11 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM16", "PM22"): (
                [1 if i == 2 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM16", "PM22"): (
                [1 if i == 1 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM16", "PM22"): (
                [1 if i == 5 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM22"): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM16", "PM22"): (
                [1 if i == 8 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 19 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM12", "PM15", "PM16", "PM23"): (
                [1 if i == 3 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 11 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM15", "PM16", "PM22", "PM23"): (
                [1 if i == 2 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM15", "PM16", "PM21", "PM22", "PM23"): (
                [1 if i == 6 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM12", "PM16", "PM21", "PM22", "PM23"): (
                [1 if i == 5 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 6 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM15", "PM21", "PM22", "PM23", "PM26"): (
                [1 if i == 1 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM21", "PM22", "PM23", "PM26"): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM16", "PM21", "PM22", "PM23", "PM26"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 16 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM11", "PM12", "PM15", "PM16", "PM21", "PM22", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 19 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM16", "PM21", "PM23", "PM26"): (
                [1 if i == 3 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 2 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM15", "PM16", "PM22", "PM23", "PM26"): (
                [1 if i == 6 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM16", "PM21", "PM22", "PM23", "PM26"): (
                [1 if i == 2 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 7 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM12", "PM16", "PM21", "PM22", "PM23", "PM25"): (
                [1 if i == 5 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 6 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM15", "PM21", "PM22", "PM23", "PM25", "PM26"): (
                [1 if i == 8 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 13 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM15", "PM21", "PM22", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 16 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM15", "PM21", "PM22", "PM24", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 15 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM15", "PM21", "PM24", "PM25", "PM26"): (
                [1 if i == 3 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 2 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM15", "PM22", "PM24", "PM25", "PM26"): (
                [1 if i == 6 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM21", "PM22", "PM24", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 18 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM21", "PM22", "PM24", "PM26"): (
                [1 if i == 1 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 7 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM11", "PM21", "PM22", "PM24", "PM25"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 6 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM21", "PM22", "PM24", "PM25", "PM26"): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 26 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM11", "PM16", "PM21", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 2 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 2 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM16", "PM22", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM16", "PM22", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 5 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 18 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM15", "PM22", "PM23", "PM24", "PM26"): (
                [1 if i == 1 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 7 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM11", "PM12", "PM15", "PM22", "PM23", "PM24", "PM25"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 16 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM11", "PM12", "PM15", "PM22", "PM24", "PM25"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM22", "PM24", "PM25"): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 26 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM16", "PM22", "PM23", "PM25"): (
                [1 if i == 3 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM15", "PM16", "PM21", "PM22", "PM23", "PM25"): (
                None, None, "specific_pms_count7", 2
            ),
            ("BM1", "BM2", "PM11", "PM16", "PM21", "PM23", "PM25"): (
                [1 if i == 2 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 6 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM16", "PM21", "PM23", "PM25", "PM26"): (
                None, None, "specific_pms_count2", 5
            ),
            ("PM12", "PM16", "PM21", "PM23", "PM25", "PM26"): (
                [1 if i == 5 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 18 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM15", "PM21", "PM23", "PM26"): (
                None, None, "specific_pms_count9", 2
            ),
            ("BM2", "PM12", "PM15", "PM21", "PM23", "PM24", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 16 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM22", "PM24", "PM26"): (
                [1 if i == 3 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 7 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM15", "PM22", "PM24", "PM25"): (
                [1 if i == 6 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM21", "PM22", "PM23", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 15 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM21", "PM23", "PM25", "PM26"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 23 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM21", "PM24", "PM25", "PM26"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 2 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM22", "PM24", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 18 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM22", "PM24", "PM26"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 7 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM22", "PM24", "PM25"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 17 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM22", "PM25"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 15 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM25"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM25",): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 18 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2",): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 0 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM21", "PM22", "PM24", "PM25"): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 6 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM16", "PM21", "PM22", "PM24", "PM25", "PM26"): (
                [1 if i == 8 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 26 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM16", "PM21", "PM22", "PM23", "PM25", "PM26"): (
                [1 if i == 2 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 13 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM16", "PM21", "PM22", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 15 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM16", "PM21", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 5 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 2 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM15", "PM22", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 18 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM11", "PM15", "PM22", "PM24", "PM25"): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 26 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM11", "PM15", "PM16", "PM22", "PM23", "PM25"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM16", "PM21", "PM22", "PM23", "PM25", "PM26"): (
                None, None, "specific_pms_count1", 3
            ),
            ("PM12", "PM15", "PM21", "PM23", "PM26"): (
                [1 if i == 8 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 16 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM12", "PM15", "PM21", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 2 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM12", "PM15", "PM22", "PM26"): (
                [1 if i == 1 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 7 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM12", "PM15", "PM22", "PM25"): (
                [1 if i == 3 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 12 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM15", "PM22", "PM23", "PM25"): (
                [1 if i == 6 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "PM11", "PM21", "PM22", "PM23", "PM25"): (
                [1 if i == 4 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 6 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM16", "PM21", "PM23", "PM26"): (
                None, None, "specific_pms_count3", 3
            ),
            ("BM2", "PM12", "PM15", "PM22", "PM23", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 23 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM15", "PM22", "PM24", "PM26"): (
                None, None, "specific_pms_count8", 2
            ),
            ("BM1", "PM12", "PM15", "PM22", "PM24", "PM25"): (
                [1 if i == 1 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 12 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM11", "PM12", "PM15", "PM22", "PM23", "PM24", "PM25"): (
                None, None, "specific_pms_count6", 3
            ),
            ("PM11", "PM15", "PM21", "PM22", "PM23", "PM24", "PM25"): (
                None, None, "specific_pms_count5", 2
            ),
            ("PM11", "PM15", "PM16", "PM21", "PM22", "PM23", "PM24", "PM25"): (
                None, None, "specific_pms_count4", 2
            ),
            ("PM11", "PM16", "PM21", "PM22", "PM23", "PM24", "PM25", "PM26"): (
                None, None, "specific_pms_count10", 3
            ),
            ("BM2", "PM12", "PM16", "PM21", "PM23", "PM24", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 17 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM15", "PM21", "PM24", "PM26"): (
                None, None, "specific_pms_count11", 2
            ),
            ("BM2", "PM11", "PM12", "PM15", "PM22", "PM23", "PM24", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 7 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM2", "PM12", "PM16", "PM21", "PM22", "PM23", "PM25", "PM26"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 15 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM12", "PM16", "PM21", "PM24", "PM25", "PM26"): (
                [1 if i == 8 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 18 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM12", "PM16", "PM21", "PM24", "PM26"): (
                [1 if i == 5 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 2 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM12", "PM15", "PM22", "PM24", "PM26"): (
                None, None, "specific_pms_count12", 2
            ),
            ("PM12", "PM15", "PM22", "PM23", "PM24", "PM26"): (
                None, None, "specific_pms_count13", 2
            ),
            ("BM1", "PM15", "PM22", "PM23", "PM24", "PM25"): (
                [1 if i == 0 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 1 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("PM15", "PM21", "PM22", "PM23", "PM24", "PM25"): (
                [1 if i == 6 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 17 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
            ("BM1", "BM2", "PM21", "PM22", "PM23", "PM25"): (
                [1 if i == 7 else 0 for i in range(len(self.robot1_actions))],
                [1 if i == 6 else 0 for i in range(len(self.robot2_actions))],
                None, 0
            ),
        }

        counter_combinations = {
            "specific_pms_count7": [
                (6, 15),
                (6, 6),
            ],
            "specific_pms_count2": [
                (7, 0),
                (7, 18),
                (7, 18),
                (7, 18),
                (7, 23),
            ],
            "specific_pms_count9": [
                (8, 13),
                (7, 0),
            ],
            "specific_pms_count1": [
                (8, 13),
                (2, 15),
                (2, 15),
            ],
            "specific_pms_count3": [
                (5, 0),
                (5, 2),
                (5, 23),
            ],
            "specific_pms_count8": [
                (1, 0),
                (8, 7),
            ],
            "specific_pms_count6": [
                (3, 16),
                (3, 1),
                (3, 1),
            ],
            "specific_pms_count5": [
                (4, 0),
                (4, 0),
            ],
            "specific_pms_count4": [
                (6, 6),
                (6, 6),
            ],
            "specific_pms_count10": [
                (0, 15),
                (2, 15),
                (2, 17),
            ],
            "specific_pms_count11": [
                (7, 2),
                (8, 2),
            ],
            "specific_pms_count12": [
                (1, 12),
                (7, 12),
            ],
            "specific_pms_count13": [
                (1, 7),
                (3, 7),
            ]
        }

        # 3. 初始化掩码（默认全1）
        mask1 = [1] * len(self.robot1_actions)
        mask2 = [1] * len(self.robot2_actions)

        # 4. 匹配字典获取掩码
        if pms_key in mask_mapping:
            r1_mask, r2_mask, counter_name, combo_len = mask_mapping[pms_key]

            # 处理普通分支（无计数器）
            if counter_name is None:
                if r1_mask is not None:
                    mask1 = r1_mask
                if r2_mask is not None:
                    mask2 = r2_mask
            # 处理带计数器的分支
            else:
                # 获取当前计数器值
                current_count = getattr(self, counter_name, 0)
                current_index = current_count % combo_len
                # 获取选中的动作组合
                selected_r1, selected_r2 = counter_combinations[counter_name][current_index]
                # 重置掩码并启用选中动作
                mask1 = [0] * len(self.robot1_actions)
                mask2 = [0] * len(self.robot2_actions)
                mask1[selected_r1] = 1
                mask2[selected_r2] = 1
                # 计数器递增
                setattr(self, counter_name, current_count + 1)
        else:
            # 未匹配到的情况，默认全1（保持原有逻辑）
            mask1 = [1] * len(self.robot1_actions)
            mask2 = [1] * len(self.robot2_actions)

        # 5. 兜底：检查并处理全0掩码
        if sum(mask1) == 0:
            mask1[1] = 1  # 强制允许空动作
            print(f"警告：mask1全0，已强制启用空动作。当前has_wafer_pms: {has_wafer_pms}")
        if sum(mask2) == 0:
            mask2[1] = 1  # 强制允许空动作
            print(f"警告：mask2全0，已强制启用空动作。当前has_wafer_pms: {has_wafer_pms}")

        # 6. 禁止关键冲突：不允许两个机器人同时执行空动作
        if mask1[0] == 1:
            mask2[0] = 0
        if mask2[0] == 1:
            mask1[0] = 0

        return [mask1, mask2]

    # 初始化分析器
    def init_system_profiler(self):
        pm_names = [pm.name for pm in self.modules]
        profiler = system_profiler(pm_names, self.modules, self.loadlock, self.robots, self.wafer_num)
        return profiler

    # 获取观测值obs
    def get_observation(self):
        # 重置失败标记（但保留外部已触发的失败）
        if not any(self.fail_flags):
            self.fail_flags = [False, False]

        # 第一步检查LL是否失败（如晶圆耗尽导致卸载失败）
        if self.loadlock.fail:
            self.fail_flags = [True, True]
            print(
            f"LL（Loadlock）操作失败：{self.loadlock.fail_reason if hasattr(self.loadlock, 'fail_reason') else '待加工晶圆耗尽'}，系统终止！")

        for pm in self.modules:
            # 检查所有模块（包括BM）的fail状态
            if pm.fail:  # 移除对非BM模块的限制
                self.fail_flags = [True, True]
                if pm.is_bm:
                    print(f"{pm.name}（Buffer模块）操作失败，系统终止！")
                else:
                    print(f"{pm.name}违反驻留时间约束，系统失败！")
                break
            # 即使PM未标记失败，也主动检查驻留时间
            if not pm.is_bm and pm.store.items.__len__() == 1:
                remaining = pm.get_residency_remaining_time()
                if remaining <= 0 and (self.env.now - pm.wafer_start_time) > (pm.process_time + pm.max_stay_time):
                    self.fail_flags = [True, True]
                    pm.fail = True     # 标记PM失败
                    print(f"{pm.name}违反驻留时间约束，系统失败！")
                    break

        # 检查机械臂卸载失败是否已触发失败标记
        if any(self.fail_flags):
            self.done = True
            # 输出最终失败终止状态
            print(
                "**********************************************************失败Terminate state!!!**********************************************************")
        else:
            # 正常状态处理（原有逻辑）
            self.done = False

        # 更新状态
        for pm in self.modules:
            wafer = pm.get_current_wafer()
            wafer_type = wafer.wafer_type if wafer else None
            self.profiler.update_modules_state(
                pm.name, pm.store.items.__len__(),
                pm.get_process_remaining_time(),
                pm.get_residency_remaining_time(),
                pm.state, pm.wafer_start_time,
                pm.get_wafer_state(),
                pm.get_wafer_processed_time(),
                pm.get_wafer_residency_time(),
                wafer_type
            )

        self.profiler.update_loadlock_state(
            self.loadlock.get_remaining_wafer_count(),
            self.loadlock.get_finished_wafer_count()
        )
        self.profiler.update_robot_state(0, self.robots[0].current_module, self.robots[0].wait_time)
        self.profiler.update_robot_state(1, self.robots[1].current_module, self.robots[1].wait_time)

        self.profiler.update_system_wafer_state()
        # 获取状态
        self.state = self.profiler.get_state()

        # 获取奖励
        self.reward, self.success_flag = self.profiler.get_reward(
            self.robot_action_flags, self.fail_flags, self.bottleneck_time
        )
        # 打印信息
        self.profiler.print_info(self.reward, self.env)

        # 检查是否完成
        if any(self.fail_flags):
            self.done = True
            print(
                "**********************************************************失败Terminate state!!!**********************************************************")
        elif self.success_flag:
            self.done = True
            print(
                "**********************************************************成功Terminate state!!!**********************************************************")
        else:
            self.done = False

        return self.state, self.reward, self.done

    # 处理器：协调两个机器人的动作执行
    def proc_handler(self):
        while True:
            yield (self.event_action)
            if self.event_action.triggered:
                self.event_action = self.env.event()

            # 重置事件和标记
            self.robot_action_flags = [False, False]
            self.event_steps = [self.env.event(), self.env.event()]

            # 启动两个机器人的动作进程
            robot1_action = self.robot1_actions[int(self.actions[0])]
            robot2_action = self.robot2_actions[int(self.actions[1])]

            self.env.process(self.execute_robot_action(0, robot1_action))
            self.env.process(self.execute_robot_action(1, robot2_action))

            # 等待两个机器人都完成动作（同步机制）
            yield self.event_steps[0] & self.event_steps[1]

            # 触发主事件
            if not self.event_hdlr.triggered:
                self.event_hdlr.succeed()

    def execute_robot_action(self, robot_id, action):
        robot = self.robots[robot_id]
        events = self.robot1_events if robot_id == 0 else self.robot2_events
        print(f"===========Robot{robot_id+1}执行动作{action}===========")
        robot.action_start_time = self.env.now
        self.robot_action_flags[robot_id] = False

        # 空动作（原地等待）
        if action == 0:
            yield self.env.timeout(0)
            self.robot_action_flags[robot_id] = True
            if not self.event_steps[robot_id].triggered:
                self.event_steps[robot_id].succeed()
            return

        # 解析动作的源和目标
        source_idx, target_idx = action
        modules_list = self.modules_list

        # 根据机器人ID映射索引到实际模块名称
        if robot_id == 0:  # robot1的模块映射
            module_map = ["LL", "PM11", "PM12", "BM1", "BM2", "PM15", "PM16"]
        else:  # robot2的模块映射
            module_map = ["PM21", "PM22", "PM23", "PM24", "PM25", "PM26", "BM1", "BM2"]

        # 检查索引有效性
        if source_idx < 0 or source_idx >= len(module_map) or target_idx < 0 or target_idx >= len(module_map):
            print(f"Robot{robot_id+1}动作索引无效")
            self.fail_flags[robot_id] = True
            self.event_steps[robot_id].succeed()
            return

        source_module = module_map[source_idx]
        target_module = module_map[target_idx]

        current_pm = None
        wafer_ut = None

        # 1. 移动到源模块 (MT)
        if not events["MT"].triggered and events["IDLE"].triggered:
            events["MT"].succeed()

        if events["MT"].triggered:
            print(f"时间:{self.env.now}\tRobot{robot_id+1}执行MT\t{robot.current_module}——>{source_module}")
            yield self.env.process(robot.move(source_module))

            # 获取当前模块对象
            if robot.current_module == "LL":
                current_pm = self.loadlock
            else:
                for pm in self.modules:
                    if pm.name == robot.current_module:
                        current_pm = pm
                        break

            events["MT"] = self.env.event()
            if not events["WT"].triggered:
                events["WT"].succeed()

        # 2. 等待 (WT)
        if events["WT"].triggered:
            wait_time = 0
            if current_pm and current_pm.name != "LL":
                wait_time = current_pm.get_process_remaining_time()

            print(f"时间:{self.env.now}\tRobot{robot_id+1}执行WT\t等待时间:{wait_time}")
            yield self.env.timeout(wait_time)
            robot.wait_time = wait_time

            events["WT"] = self.env.event()

            if not events["UT"].triggered:
                events["UT"].succeed()

        # 3. 卸载晶圆 (UT) - 关键修复：使用yield获取实际晶圆对象
        if events["UT"].triggered:
            print(f"时间:{self.env.now}\tRobot{robot_id+1}执行UT\t从{robot.current_module}卸载晶圆")
            if robot.current_module == "LL" :
                # 从LL卸载新晶圆（根据交替逻辑）
                wafer_ut = current_pm.unload()
            else:
                # 从PM卸载已加工晶圆（使用yield获取实际对象）
                wafer_ut = yield self.env.process(current_pm.unload())  # 关键修复

            # 检查卸载是否失败
            if current_pm.fail or wafer_ut is None:
                self.fail_flags[robot_id] = True
                events["UT"] = self.env.event()
                self.event_steps[robot_id].succeed()
                return

            yield self.env.process(robot.load(wafer_ut))
            events["UT"] = self.env.event()

            if not events["CT"].triggered:
                events["CT"].succeed()

        # 4. 携带晶圆 (CT)
        if events["CT"].triggered and wafer_ut:
            print(f"时间:{self.env.now}\tRobot{robot_id+1}执行CT\t携带晶圆W{wafer_ut.id}")
            yield self.env.process(robot.move(target_module))
            if robot.fail:
                self.fail_flags[robot_id] = True
                events["CT"] = self.env.event()
                self.event_steps[robot_id].succeed()
                return

            events["CT"] = self.env.event()
            if not events["LT"].triggered:
                events["LT"].succeed()

        # 5. 移动到目标模块并加载晶圆 (LT)
        if events["LT"].triggered and wafer_ut:
            print(f"时间:{self.env.now}\tRobot{robot_id+1}执行LT\t从{robot.current_module}移动到{target_module}")
            yield self.env.process(robot.move(target_module))

            # 获取目标模块对象
            target_pm = None
            if robot.current_module == "LL":
                target_pm = self.loadlock
            else:
                for pm in self.modules:
                    if pm.name == robot.current_module:
                        target_pm = pm
                        break

            if not target_pm:
                print(f"时间:{self.env.now}\tRobot{robot_id+1}目标模块{target_module}不存在")
                self.fail_flags[robot_id] = True
                events["LT"] = self.env.event()
                self.event_steps[robot_id].succeed()
                return

            # 卸载机器人上的晶圆到目标模块
            wafer_lt = yield self.env.process(robot.unload())
            if robot.fail or wafer_lt is None:
                self.fail_flags[robot_id] = True
                events["LT"] = self.env.event()
                self.event_steps[robot_id].succeed()
                return

            # 加载晶圆到目标模块
            target_pm.load(wafer_lt)
            if target_pm.fail:
                self.fail_flags[robot_id] = True
                events["LT"] = self.env.event()
                self.event_steps[robot_id].succeed()
                return

            events["LT"] = self.env.event()
            if not events["IDLE"].triggered:
                events["IDLE"].succeed()

        # 计算动作时间
        robot.take_action_time = self.env.now - robot.action_start_time
        self.robot_action_flags[robot_id] = True

        # 触发步骤完成事件
        if not self.event_steps[robot_id].triggered:
            self.event_steps[robot_id].succeed()

# 解析参数
def parse_args():
    parser = argparse.ArgumentParser(description='双机器人晶圆加工环境参数')
    # 动作和模块配置
    parser.add_argument('--robot_actions', type=list, default=["MT", "WT", "UT", "CT", "LT", "IDLE"])
    parser.add_argument('--action_dim', type=int, default=8)  # 最大动作维度
    # 时间参数
    parser.add_argument('--unload_time_LL', type=int, default=10)
    parser.add_argument('--work_time', type=int, default=5)
    parser.add_argument('--move_time', type=int, default=2)
    # 工艺时间参数（示例值）
    parser.add_argument('--process_residency_time_dict', type=dict, default={})
    parser.add_argument('--process_residency_time_list', type=list, default=[])
    parser.add_argument('--max_stay_time_dict', type=dict, default={})
    parser.add_argument('--max_stay_time_list', type=list, default=[])
    parser.add_argument('--process_time_dict', type=dict, default={})
    parser.add_argument('--process_time_list', type=list, default=[])
    parser.add_argument('--wait_time_dict', type=dict, default={})
    parser.add_argument('--wait_time_list', type=list, default=[])

    return parser.parse_args()


# 测试环境
if __name__ == "__main__":
    args = parse_args()
    env = Environment(args, wafer_num=30, wafer_type_distribution=[ 1/3, 1/3, 1/3 ])
    state = env.reset()

    # 定义有效动作序列：每个元素是 [robot1索引, robot2索引] 的组合
    # 确保索引不超过机器人动作空间的长度
    action_sequence = [
        [1, 0],
        [4, 0],
        [8, 0],
        [0, 11],
        [2, 0],
        [1, 0],
        [5, 0],
        [4, 0],
        [8, 19],
        [3, 11],
        [2, 1],
        [6, 0],
        [5, 6],
        [1, 0],
        [4, 0],
        [0, 16],
        [7, 19],
        [3, 2],
        [6, 1],
        [2, 7],
        [5, 6],
        [8, 13],
        [0, 16],
        [7, 15],
        [3, 2],
        [6, 1],
        [7, 18],
        [1, 7],
        [7, 6],
        [4, 26],
        [8, 13],
        [0, 15],
        [2, 2],
        [7, 0],
        [5, 18],
        [1, 7],
        [7, 16],
        [7, 0],
        [4, 26],
        [3, 1],
        [6, 15],
        [2, 6],
        [7, 0],
        [5, 18],
        [8, 13],
        [7, 16],
        [7, 2],
        [1, 0],
        [3, 7],
        [6, 1],
        [4, 6],
        [8, 26],
        [2, 13],
        [7, 15],
        [5, 2],
        [7, 18],
        [1, 7],
        [3, 16],
        [4, 26],
        [7, 1],
        [6, 6],
        [2, 15],
        [7, 18],
        [5, 0],
        [7, 0],
        [8, 16],
        [7, 2],
        [1, 7],
        [3, 12],
        [6, 1],
        [4, 6],
        [2, 15],
        [7, 18],
        [5, 2],
        [7, 23],
        [8, 7],
        [1, 12],
        [3, 1],
        [4, 0],
        [6, 6],
        [2, 15],
        [7, 17],
        [7, 18],
        [5, 23],
        [8, 2],
        [1, 12],
        [7, 7],
        [3, 1],
        [4, 0],
        [6, 6],
        [2, 17],
        [7, 15],
        [7, 23],
        [8, 18],
        [5, 2],
        [7, 12],
        [3, 7],
        [0, 1],
        [6, 17],
        [7, 6],
        [7, 15],
        [0, 23],
        [0, 2],
        [7, 18],
        [0, 7],
        [0, 17],
        [7, 15],
        [7, 0],
        [0, 18],
        [7, 0],
    ]

    for step, (a1, a2) in enumerate(action_sequence, 1):
        # 检查动作索引是否有效
        if a1 < 0 or a1 >= len(env.robot1_actions):
            print(f"无效的robot1动作索引: {a1}，有效范围0-{len(env.robot1_actions)-1}")
            continue
        if a2 < 0 or a2 >= len(env.robot2_actions):
            print(f"无效的robot2动作索引: {a2}，有效范围0-{len(env.robot2_actions)-1}")
            continue

        print(
            f"\n第{step}步：执行动作 - robot1索引={a1}（动作{env.robot1_actions[a1]}）, robot2索引={a2}（动作{env.robot2_actions[a2]}）")
        next_state, reward, done = env.step([a1, a2])
        print(env.get_mask())
        if done:
            break

    print("\n所有指定动作执行完毕！")