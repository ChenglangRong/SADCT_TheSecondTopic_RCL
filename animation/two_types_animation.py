import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch  # 支持分段箭头绘制

# 全局配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 闪烁配置常量（可直接修改，无需调整核心逻辑） ----------------------
WAFER_COLOR_GRAY = '#808080'  # 晶圆默认灰色（LL工位/非闪烁状态）
WAFER_COLOR_GOLD = '#FFD700'  # 晶圆闪烁金色（PM工位加工状态）
FLASH_INTERVAL = 500  # 闪烁循环间隔（毫秒），数值越小闪烁越快

# ---------------------- 全局状态定义 ----------------------
# 工位坐标常量（完全对齐参考图布局和配色）
STATIONS = {
    'PM1': (-7, 3.5, 'yellow'), 'PM2': (-3, 3.5, 'lightblue'),
    'PM9': (-3, -3.5, 'lightblue'), 'PM10': (-7, -3.5, 'yellow'),
    'LLA': (-9, 0.75, 'white'), 'LLB': (-9, -1.25, 'white'),
    'PM3': (1, 3.75, 'white'), 'PM4': (4, 3.75, 'white'),
    'PM5': (6.25, 1.5, 'white'), 'PM6': (6.25, -1.5, 'white'),
    'PM7': (4, -3.75, 'white'), 'PM8': (1, -3.75, 'white'),
    'LLC': (-1.25, 1.25, 'orange'), 'LLD': (-1.25, -1.25, 'orange')
}

# 机械臂配置
ROBOTS = {
    1: {'joint': (-5, 0), 'color': 'black', 'initial_target': 'LLA'},
    2: {'joint': (2.5, 0), 'color': 'black', 'initial_target': 'LLC'}
}

BASE_LENGTH = 2  # 机械臂基础长度

# 全局状态管理器（升级闪烁状态：支持多晶圆独立闪烁，记录<晶圆:定时器>映射）
state_manager = {
    'robots': {}, 'station_wafers': {}, 'task_queue': [],
    'frames_per_step': 120, 'current_step': -1, 'total_steps': 0,
    'animation_running': False, 'step_animation': None,
    'root': None, 'canvas': None, 'status_label': None, 'progress_var': None,
    'arrows': [],  # 存储原始任务箭头（普通样式）
    'process_arrows': [],  # 存储加工路径箭头（特殊样式）
    'arrows_visible': True,  # 控制普通箭头显示/隐藏
    'show_process_path': False,  # 控制加工路径显示/隐藏
    'wafer_flash_timers': {}  # 键：晶圆对象，值：对应闪烁定时器（支持多晶圆独立闪烁）
}

# ---------------------- 新增：加工路径定制化配置 ----------------------
# 定制化路径节点映射（用于判断是否为指定特殊路径）
CUSTOM_PATHS = {
    'RED_PATH': ['LLC', 'PM3', 'PM4', 'PM5', 'LLD'],  # 纯红色路径节点
    'GREEN_PATH': ['LLC', 'PM8', 'PM7', 'PM6', 'LLD'],  # 纯绿色路径节点
    'DOUBLE_ARROW': [('LLD', 'LLA')]  # 红绿双箭头路径
}


def is_in_custom_chain(source, target, chain_nodes):
    """判断source→target是否为指定链式路径的一环（如LLC→PM3是RED_PATH的一环）"""
    try:
        s_idx = chain_nodes.index(source)
        t_idx = chain_nodes.index(target)
        return t_idx == s_idx + 1  # 确保是链式下一个节点
    except ValueError:
        return False


def is_double_arrow_path(source, target):
    """判断是否为LLD→LLA双箭头路径"""
    return (source, target) in CUSTOM_PATHS['DOUBLE_ARROW']


# ---------------------- 核心升级：晶圆灰金循环闪烁工具函数 ----------------------
def is_pm_station(station_name):
    """判断是否为PM加工工位（PM开头为加工工位，LL开头为非加工工位）"""
    return station_name.startswith('PM')


def start_wafer_flash(wafer):
    """
    启动晶圆灰金循环闪烁（灰色→金色→灰色）
    只要晶圆未被停止/移出PM工位，将持续循环闪烁
    """
    # 若晶圆已在闪烁，先停止原有循环，避免重复定时器
    if wafer in state_manager['wafer_flash_timers']:
        stop_wafer_flash(wafer)

    # 初始化闪烁状态：当前为灰色（默认状态）
    current_is_gray = [True]

    def flash_cycle():
        """灰金循环核心逻辑：切换晶圆填充色，循环执行"""
        # 若晶圆已被移除闪烁列表（如被抓取/移出PM），停止循环
        if wafer not in state_manager['wafer_flash_timers']:
            return
        # 切换颜色：灰色↔金色
        if current_is_gray[0]:
            wafer.set_facecolor(WAFER_COLOR_GOLD)  # 灰色→金色
        else:
            wafer.set_facecolor(WAFER_COLOR_GRAY)  # 金色→灰色
        current_is_gray[0] = not current_is_gray[0]
        # 刷新画布生效
        state_manager['canvas'].draw()
        # 定时执行下一次颜色切换，更新定时器（保证循环不中断）
        state_manager['wafer_flash_timers'][wafer] = state_manager['root'].after(
            FLASH_INTERVAL, flash_cycle
        )

    # 注册晶圆并启动第一次闪烁
    state_manager['wafer_flash_timers'][wafer] = state_manager['root'].after(
        FLASH_INTERVAL, flash_cycle
    )


def stop_wafer_flash(wafer=None):
    """
    停止晶圆闪烁：恢复灰色，清除定时器
    - 传wafer：停止指定晶圆的闪烁
    - 不传wafer：停止所有晶圆的闪烁（如程序关闭/批量清理）
    """
    if wafer:
        # 停止指定晶圆的闪烁
        if wafer in state_manager['wafer_flash_timers']:
            # 取消定时器
            state_manager['root'].after_cancel(state_manager['wafer_flash_timers'][wafer])
            del state_manager['wafer_flash_timers'][wafer]
            # 恢复默认灰色
            wafer.set_facecolor(WAFER_COLOR_GRAY)
            state_manager['canvas'].draw()
    else:
        # 停止所有晶圆的闪烁（批量清理）
        for w in list(state_manager['wafer_flash_timers'].keys()):
            stop_wafer_flash(w)


# ---------------------- 绘图工具函数 ----------------------
def draw_station(ax, x, y, color, label):
    """绘制正八边形工位+内部同心圆（替换原六边形，核心修改函数）"""
    # 正八边形顶点计算：外接圆半径1.0，8个顶点均匀分布
    octagon_vertices = []
    for i in range(8):
        angle = np.pi / 4 * i  # 八边形每个顶点的角度（45度间隔）
        vx = x + 1.0 * np.cos(angle)
        vy = y + 1.0 * np.sin(angle)
        octagon_vertices.append((vx, vy))
    # 绘制正八边形（外框）
    octagon = plt.Polygon(octagon_vertices, edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(octagon)
    # 绘制八边形内部同心圆（与八边形同中心，半径0.6，白色边框+同底色填充）
    inner_circle = plt.Circle((x, y), 0.6, facecolor=color, edgecolor='black', linewidth=1.2)
    ax.add_patch(inner_circle)
    # 绘制工位标签（居中显示，加粗）
    ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold', color='black')


def draw_task_arrow(ax, source_name, target_name, color='red', linestyle='-', linewidth=1.5):
    """绘制工位间的任务流向箭头（适配八边形，修复norm未定义错误）"""
    x1, y1, _ = STATIONS[source_name]
    x2, y2, _ = STATIONS[target_name]
    # 计算箭头起点终点（适配八边形尺寸，偏移量1.2避免与图形重叠）
    dir_x = x2 - x1
    dir_y = y2 - y1
    # 修复核心：先计算norm，再判断是否为0（避免引用前置）
    norm = np.sqrt(dir_x ** 2 + dir_y ** 2)
    norm = norm if norm != 0 else 1  # 非零保护，防止除以0
    start_x = x1 + dir_x / norm * 1.2
    start_y = y1 + dir_y / norm * 1.2
    end_x = x2 - dir_x / norm * 1.2
    end_y = y2 - dir_y / norm * 1.2

    arrow = ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                     head_width=0.3, head_length=0.3,
                     fc=color, ec=color, linestyle=linestyle, linewidth=linewidth,
                     length_includes_head=True, alpha=0.7)
    state_manager['arrows'].append(arrow)
    return arrow


def draw_process_arrow(ax, source_name, target_name, robot_id):

    x1, y1, _ = STATIONS[source_name]
    x2, y2, _ = STATIONS[target_name]
    # 计算箭头起点终点（与普通箭头保持一致，避免重叠）
    dir_x = x2 - x1
    dir_y = y2 - y1
    norm = np.sqrt(dir_x ** 2 + dir_y ** 2)
    norm = norm if norm != 0 else 1  # 非零判断，防止除以0
    start_x = x1 + dir_x / norm * 1.2
    start_y = y1 + dir_y / norm * 1.2
    end_x = x2 - dir_x / norm * 1.2
    end_y = y2 - dir_y / norm * 1.2
    # 计算箭头方向垂直偏移量（用于双箭头并排，间距0.2）
    perp_x = -dir_y / norm * 0.2  # 垂直于箭头方向的x偏移
    perp_y = dir_x / norm * 0.2  # 垂直于箭头方向的y偏移

    # 定制规则1：LLD→LLA 红绿并排双箭头（红上绿下）
    if is_double_arrow_path(source_name, target_name):
        # 红色箭头（向上偏移）
        red_arrow = ax.arrow(start_x + perp_x, start_y + perp_y,
                             end_x - start_x, end_y - start_y,
                             head_width=0.3, head_length=0.3,
                             fc='red', ec='red', linewidth=1.8,
                             length_includes_head=True, alpha=0.8)
        # 绿色箭头（向下偏移）
        green_arrow = ax.arrow(start_x - perp_x, start_y - perp_y,
                               end_x - start_x, end_y - start_y,
                               head_width=0.3, head_length=0.3,
                               fc='green', ec='green', linewidth=1.8,
                               length_includes_head=True, alpha=0.8)
        state_manager['process_arrows'].extend([red_arrow, green_arrow])
        return

    # 定制规则2：LLC→PM3→PM4→PM5→LLD 纯红色箭头
    if is_in_custom_chain(source_name, target_name, CUSTOM_PATHS['RED_PATH']):
        red_arrow = ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                             head_width=0.3, head_length=0.3,
                             fc='red', ec='red', linewidth=1.8,
                             length_includes_head=True, alpha=0.8)
        state_manager['process_arrows'].append(red_arrow)
        return

    # 定制规则3：LLC→PM8→PM7→PM6→LLD 纯绿色箭头
    if is_in_custom_chain(source_name, target_name, CUSTOM_PATHS['GREEN_PATH']):
        green_arrow = ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                               head_width=0.3, head_length=0.3,
                               fc='green', ec='green', linewidth=1.8,
                               length_includes_head=True, alpha=0.8)
        state_manager['process_arrows'].append(green_arrow)
        return

    # 原规则：非定制路径，按机械臂id绘制
    arrow_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    segment1_ratio = 0.6  # 上半部分比例
    segment1_end_x = start_x + (end_x - start_x) * segment1_ratio
    segment1_end_y = start_y + (end_y - start_y) * segment1_ratio

    if robot_id == 1:
        # 机械臂1加工路径：整体黑色箭头（含头部）
        arrow = ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                         head_width=0.3, head_length=0.3,
                         fc='black', ec='black', linewidth=1.8,
                         length_includes_head=True, alpha=0.8)
        state_manager['process_arrows'].append(arrow)
    else:
        # 机械臂2加工路径：分段绘制（上红下绿+绿色头部）
        arrow_upper = FancyArrowPatch((start_x, start_y), (segment1_end_x, segment1_end_y),
                                      color='red', linewidth=1.8, arrowstyle='-', alpha=0.8)
        ax.add_patch(arrow_upper)
        arrow_lower = ax.arrow(segment1_end_x, segment1_end_y,
                               end_x - segment1_end_x, end_y - segment1_end_y,
                               head_width=0.3, head_length=0.3,
                               fc='green', ec='green', linewidth=1.8,
                               length_includes_head=True, alpha=0.8)
        state_manager['process_arrows'].extend([arrow_upper, arrow_lower])


def draw_robot_base(ax, robot_id):

    joint_x, joint_y = ROBOTS[robot_id]['joint']
    # 外圆（底座）
    robot_circle = plt.Circle((joint_x, joint_y), 3, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(robot_circle)
    # 内圆（关节点）
    robot_joint = plt.Circle((joint_x, joint_y), 0.3, facecolor='black')
    ax.add_patch(robot_joint)
    # 机械臂底座标签
    ax.text(joint_x, joint_y - 6, f'Cluster Tools {robot_id}', ha='center', va='center', fontsize=10)


def create_wafer(ax):
    """创建新晶圆（默认灰色，匹配参考图样式）"""
    wafer = plt.Circle(
        (0, 0), 0.5,
        facecolor=WAFER_COLOR_GRAY, edgecolor='white',
        linewidth=0.5, visible=False, alpha=1.0
    )
    ax.add_patch(wafer)
    return wafer


def init_robot(ax, robot_id):
    """初始化机械臂（带参考图的爪状结构）"""
    # 机械臂连杆
    link, = ax.plot([], [], color=ROBOTS[robot_id]['color'], linewidth=4)
    joint_x, joint_y = ROBOTS[robot_id]['joint']
    target_x, target_y, _ = STATIONS[ROBOTS[robot_id]['initial_target']]

    # 计算初始方向
    dir_x, dir_y = target_x - joint_x, target_y - joint_y
    norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
    end_x, end_y = joint_x + dir_x / norm * BASE_LENGTH, joint_y + dir_y / norm * BASE_LENGTH

    # 绘制机械臂爪（参考图的三角形爪状结构）
    claw_angle = np.pi / 8  # 爪的张开角度
    claw_length = 0.6
    # 左爪
    claw1_x = end_x + claw_length * np.cos(np.arctan2(dir_y, dir_x) + claw_angle)
    claw1_y = end_y + claw_length * np.sin(np.arctan2(dir_y, dir_x) + claw_angle)
    # 右爪
    claw2_x = end_x + claw_length * np.cos(np.arctan2(dir_y, dir_x) - claw_angle)
    claw2_y = end_y + claw_length * np.sin(np.arctan2(dir_y, dir_x) - claw_angle)
    # 爪的连线
    claw, = ax.plot([end_x, claw1_x, end_x, claw2_x], [end_y, claw1_y, end_y, claw2_y],
                    color=ROBOTS[robot_id]['color'], linewidth=3)

    # 设置连杆数据
    link.set_data([joint_x, end_x], [joint_y, end_y])
    state_manager['robots'][robot_id] = {
        'link': link, 'claw': claw, 'active_wafer': None, 'last_pos': (end_x, end_y)
    }
    return link


def draw_all_task_arrows(ax, task_matrix):
    """根据任务矩阵绘制所有任务流向箭头（参考图配色，原逻辑不变）"""
    color_map = {'robot1': 'red', 'robot2': 'green'}
    for step_idx, (task1, task2) in enumerate(task_matrix):
        # 机械臂1任务箭头（红色）
        if task1 != 0 and len(task1) == 2:
            draw_task_arrow(ax, task1[0], task1[1], color=color_map['robot1'])
        # 机械臂2任务箭头（绿色）
        if task2 != 0 and len(task2) == 2:
            draw_task_arrow(ax, task2[0], task2[1], color=color_map['robot2'])


def draw_all_process_arrows(ax, task_matrix):
    """绘制所有加工路径箭头（核心：调用修改后的draw_process_arrow，应用定制规则）"""
    # 先清空原有加工路径箭头，避免重复绘制
    for arrow in state_manager['process_arrows']:
        arrow.remove()
    state_manager['process_arrows'].clear()

    # 遍历任务矩阵绘制加工路径
    for step_idx, (task1, task2) in enumerate(task_matrix):
        if task1 != 0 and len(task1) == 2:
            draw_process_arrow(ax, task1[0], task1[1], robot_id=1)
        if task2 != 0 and len(task2) == 2:
            draw_process_arrow(ax, task2[0], task2[1], robot_id=2)


# ---------------------- 箭头控制函数 ----------------------
def toggle_arrows_visibility():
    """切换普通任务箭头的显示/隐藏状态（原函数不变，无任何修改）"""
    state_manager['arrows_visible'] = not state_manager['arrows_visible']

    # 更新所有普通箭头的可见性
    for arrow in state_manager['arrows']:
        arrow.set_visible(state_manager['arrows_visible'])

    # 若当前显示加工路径，普通箭头隐藏不影响加工路径
    if state_manager['show_process_path']:
        for arrow in state_manager['process_arrows']:
            arrow.set_visible(True)

    # 更新按钮文本和状态提示
    btn_text = "🔴 机械臂箭头" if state_manager['arrows_visible'] else "🟢 机械臂箭头"
    toggle_btn.config(text=btn_text)
    state_manager['status_label'].config(
        text=f"✅ 机械臂任务箭头已{'显示' if state_manager['arrows_visible'] else '隐藏'} | PM工位晶圆灰金循环闪烁 | 按回车/点按钮执行下一步"
    )

    # 刷新画布
    state_manager['canvas'].draw()


def toggle_process_path():
    """切换晶圆加工路径显示/隐藏（原函数不变，微调状态提示）"""
    state_manager['show_process_path'] = not state_manager['show_process_path']
    ax = state_manager['canvas'].figure.axes[0]  # 获取绘图轴对象

    if state_manager['show_process_path']:
        # 显示加工路径：隐藏普通箭头 → 绘制/显示加工路径箭头
        for arrow in state_manager['arrows']:
            arrow.set_visible(False)
        # 首次显示时绘制加工路径，后续直接显示
        if not state_manager['process_arrows']:
            draw_all_process_arrows(ax, state_manager['task_queue'])
        else:
            for arrow in state_manager['process_arrows']:
                arrow.set_visible(True)
        # 更新按钮文本和状态提示（适配灰金闪烁）
        process_btn.config(text="🟡 加工路径")
        state_manager['status_label'].config(
            text="✅ 已显示晶圆加工路径 | LLD→LLA=红绿双箭 | LLC→PM3→PM5=红 | LLC→PM8→PM6=绿 | PM工位晶圆灰金循环闪烁"
        )
    else:
        # 隐藏加工路径：隐藏加工路径箭头 → 恢复普通箭头显示（按原有状态）
        for arrow in state_manager['process_arrows']:
            arrow.set_visible(False)
        for arrow in state_manager['arrows']:
            arrow.set_visible(state_manager['arrows_visible'])
        # 更新按钮文本和状态提示
        process_btn.config(text="🔵 加工路径")
        state_manager['status_label'].config(
            text=f"✅ 已隐藏加工路径 | 机械臂箭头{'显示' if state_manager['arrows_visible'] else '隐藏'} | PM工位晶圆灰金循环闪烁 | 按回车/点按钮执行下一步"
        )

    # 刷新画布
    state_manager['canvas'].draw()


# ---------------------- 机械臂动作执行函数（核心升级：灰金闪烁+持久化控制） ----------------------
def execute_robot_action(robot_id, task, frame_in_step, ax):
    """执行单个机械臂动作（适配爪状机械臂，升级晶圆灰金循环闪烁逻辑）"""
    robot_state = state_manager['robots'][robot_id]
    joint_x, joint_y = ROBOTS[robot_id]['joint']

    if task == 0:  # 原地不动
        if robot_state['last_pos']:
            last_x, last_y = robot_state['last_pos']
            # 更新连杆
            robot_state['link'].set_data([joint_x, last_x], [joint_y, last_y])
            # 更新爪
            dir_x, dir_y = last_x - joint_x, last_y - joint_y
            norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
            claw_angle = np.pi / 8
            claw_length = 0.6
            claw1_x = last_x + dir_x / norm * claw_length * np.cos(claw_angle) - dir_y / norm * claw_length * np.sin(
                claw_angle)
            claw1_y = last_y + dir_y / norm * claw_length * np.cos(claw_angle) + dir_x / norm * claw_length * np.sin(
                claw_angle)
            claw2_x = last_x + dir_x / norm * claw_length * np.cos(-claw_angle) - dir_y / norm * claw_length * np.sin(
                -claw_angle)
            claw2_y = last_y + dir_y / norm * claw_length * np.cos(-claw_angle) + dir_x / norm * claw_length * np.sin(
                -claw_angle)
            robot_state['claw'].set_data([last_x, claw1_x, last_x, claw2_x], [last_y, claw1_y, last_y, claw2_y])
        return

    # 执行搬运任务
    source_name, target_name = task
    source_x, source_y, _ = STATIONS[source_name]
    target_x, target_y, _ = STATIONS[target_name]

    # 阶段1: 初始指向源点(0-10帧)
    if frame_in_step <= 10:
        dir_x, dir_y = source_x - joint_x, source_y - joint_y
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
        end_x, end_y = joint_x + dir_x / norm * BASE_LENGTH, joint_y + dir_y / norm * BASE_LENGTH
        # 更新连杆
        robot_state['link'].set_data([joint_x, end_x], [joint_y, end_y])
        # 更新爪
        claw_angle = np.pi / 8
        claw_length = 0.6
        claw1_x = end_x + dir_x / norm * claw_length * np.cos(claw_angle) - dir_y / norm * claw_length * np.sin(
            claw_angle)
        claw1_y = end_y + dir_y / norm * claw_length * np.cos(claw_angle) + dir_x / norm * claw_length * np.sin(
            claw_angle)
        claw2_x = end_x + dir_x / norm * claw_length * np.cos(-claw_angle) - dir_y / norm * claw_length * np.sin(
            -claw_angle)
        claw2_y = end_y + dir_y / norm * claw_length * np.cos(-claw_angle) + dir_x / norm * claw_length * np.sin(
            -claw_angle)
        robot_state['claw'].set_data([end_x, claw1_x, end_x, claw2_x], [end_y, claw1_y, end_y, claw2_y])

        robot_state['last_pos'] = (end_x, end_y)
        if source_name in state_manager['station_wafers']:
            state_manager['station_wafers'][source_name].set_visible(True)

    # 阶段2: 伸长到源点(11-30帧)
    elif 10 < frame_in_step <= 30:
        t = (frame_in_step - 10) / 20
        end_x = np.interp(t, [0, 1], [
            joint_x + (source_x - joint_x) / np.linalg.norm([source_x - joint_x, source_y - joint_y]) * BASE_LENGTH,
            source_x])
        end_y = np.interp(t, [0, 1], [
            joint_y + (source_y - joint_y) / np.linalg.norm([source_x - joint_x, source_y - joint_y]) * BASE_LENGTH,
            source_y])
        # 更新连杆
        robot_state['link'].set_data([joint_x, end_x], [joint_y, end_y])
        # 更新爪
        dir_x, dir_y = end_x - joint_x, end_y - joint_y
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
        claw_angle = np.pi / 8
        claw_length = 0.6
        claw1_x = end_x + dir_x / norm * claw_length * np.cos(claw_angle) - dir_y / norm * claw_length * np.sin(
            claw_angle)
        claw1_y = end_y + dir_y / norm * claw_length * np.cos(claw_angle) + dir_x / norm * claw_length * np.sin(
            claw_angle)
        claw2_x = end_x + dir_x / norm * claw_length * np.cos(-claw_angle) - dir_y / norm * claw_length * np.sin(
            -claw_angle)
        claw2_y = end_y + dir_y / norm * claw_length * np.cos(-claw_angle) + dir_x / norm * claw_length * np.sin(
            -claw_angle)
        robot_state['claw'].set_data([end_x, claw1_x, end_x, claw2_x], [end_y, claw1_y, end_y, claw2_y])

        robot_state['last_pos'] = (end_x, end_y)
        if frame_in_step == 30:  # 到达源点抓取晶圆：停止源点的闪烁（如果是PM工位）
            robot_state['active_wafer'] = state_manager['station_wafers'].pop(source_name) if source_name in \
                                                                                              state_manager[
                                                                                                  'station_wafers'] else create_wafer(
                ax)
            robot_state['active_wafer'].center = (source_x, source_y)
            robot_state['active_wafer'].set_visible(True)
            # 抓取晶圆时，停止源工位的闪烁（无论源工位是否为PM）
            stop_wafer_flash(robot_state['active_wafer'])

    # 阶段3: 抓取停留(31-40帧)
    elif 30 < frame_in_step <= 40:
        # 更新连杆
        robot_state['link'].set_data([joint_x, source_x], [joint_y, source_y])
        # 更新爪
        dir_x, dir_y = source_x - joint_x, source_y - joint_y
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
        claw_angle = np.pi / 8
        claw_length = 0.6
        claw1_x = source_x + dir_x / norm * claw_length * np.cos(claw_angle) - dir_y / norm * claw_length * np.sin(
            claw_angle)
        claw1_y = source_y + dir_y / norm * claw_length * np.cos(claw_angle) + dir_x / norm * claw_length * np.sin(
            claw_angle)
        claw2_x = source_x + dir_x / norm * claw_length * np.cos(-claw_angle) - dir_y / norm * claw_length * np.sin(
            -claw_angle)
        claw2_y = source_y + dir_y / norm * claw_length * np.cos(-claw_angle) + dir_x / norm * claw_length * np.sin(
            -claw_angle)
        robot_state['claw'].set_data([source_x, claw1_x, source_x, claw2_x], [source_y, claw1_y, source_y, claw2_y])

        robot_state['last_pos'] = (source_x, source_y)
        if robot_state['active_wafer']:
            robot_state['active_wafer'].center = (source_x, source_y)

    # 阶段4: 移动到目标点(41-80帧)
    elif 40 < frame_in_step <= 80:
        t = (frame_in_step - 40) / 40
        end_x, end_y = np.interp(t, [0, 1], [source_x, target_x]), np.interp(t, [0, 1], [source_y, target_y])
        # 更新连杆
        robot_state['link'].set_data([joint_x, end_x], [joint_y, end_y])
        # 更新爪
        dir_x, dir_y = end_x - joint_x, end_y - joint_y
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
        claw_angle = np.pi / 8
        claw_length = 0.6
        claw1_x = end_x + dir_x / norm * claw_length * np.cos(claw_angle) - dir_y / norm * claw_length * np.sin(
            claw_angle)
        claw1_y = end_y + dir_y / norm * claw_length * np.cos(claw_angle) + dir_x / norm * claw_length * np.sin(
            claw_angle)
        claw2_x = end_x + dir_x / norm * claw_length * np.cos(-claw_angle) - dir_y / norm * claw_length * np.sin(
            -claw_angle)
        claw2_y = end_y + dir_y / norm * claw_length * np.cos(-claw_angle) + dir_x / norm * claw_length * np.sin(
            -claw_angle)
        robot_state['claw'].set_data([end_x, claw1_x, end_x, claw2_x], [end_y, claw1_y, end_y, claw2_y])

        robot_state['last_pos'] = (end_x, end_y)
        if robot_state['active_wafer']:
            robot_state['active_wafer'].center = (end_x, end_y)

    # 阶段5: 放置晶圆(81-90帧) - 核心升级：PM工位启动持久化灰金闪烁，LL工位静态灰色
    elif 80 < frame_in_step <= 90:
        # 更新连杆
        robot_state['link'].set_data([joint_x, target_x], [joint_y, target_y])
        # 更新爪
        dir_x, dir_y = target_x - joint_x, target_y - joint_y
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
        claw_angle = np.pi / 8
        claw_length = 0.6
        claw1_x = target_x + dir_x / norm * claw_length * np.cos(claw_angle) - dir_y / norm * claw_length * np.sin(
            claw_angle)
        claw1_y = target_y + dir_y / norm * claw_length * np.cos(claw_angle) + dir_x / norm * claw_length * np.sin(
            claw_angle)
        claw2_x = target_x + dir_x / norm * claw_length * np.cos(-claw_angle) - dir_y / norm * claw_length * np.sin(
            -claw_angle)
        claw2_y = target_y + dir_y / norm * claw_length * np.cos(-claw_angle) + dir_x / norm * claw_length * np.sin(
            -claw_angle)
        robot_state['claw'].set_data([target_x, claw1_x, target_x, claw2_x], [target_y, claw1_y, target_y, claw2_y])

        robot_state['last_pos'] = (target_x, target_y)
        if robot_state['active_wafer']:
            robot_state['active_wafer'].center = (target_x, target_y)
            # LLA工位特殊处理：隐藏晶圆，停止闪烁
            if target_name == 'LLA':
                robot_state['active_wafer'].set_visible(False)
                stop_wafer_flash(robot_state['active_wafer'])
            # 将晶圆绑定到目标工位
            state_manager['station_wafers'][target_name] = robot_state['active_wafer']
            robot_state['active_wafer'] = None

            # 核心逻辑：判断目标工位类型，控制闪烁
            if is_pm_station(target_name) and target_name != 'LLA':
                # PM工位：启动灰金循环闪烁，只要晶圆在PM上就持续闪烁
                start_wafer_flash(state_manager['station_wafers'][target_name])
            else:
                # LL工位：保持灰色静态，停止闪烁（若之前有闪烁）
                stop_wafer_flash(state_manager['station_wafers'][target_name])

    # 阶段6: 缩回初始长度(91-120帧)
    elif 90 < frame_in_step <= 120:
        t = (frame_in_step - 90) / 30
        dir_x, dir_y = target_x - joint_x, target_y - joint_y
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
        end_x, end_y = np.interp(t, [0, 1], [target_x, joint_x + dir_x / norm * BASE_LENGTH]), np.interp(t, [0, 1],
                                                                                                         [target_y,
                                                                                                          joint_y + dir_y / norm * BASE_LENGTH])
        # 更新连杆
        robot_state['link'].set_data([joint_x, end_x], [joint_y, end_y])
        # 更新爪
        dir_x, dir_y = end_x - joint_x, end_y - joint_y
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2) if dir_x ** 2 + dir_y ** 2 != 0 else 1
        claw_angle = np.pi / 8
        claw_length = 0.6
        claw1_x = end_x + dir_x / norm * claw_length * np.cos(claw_angle) - dir_y / norm * claw_length * np.sin(
            claw_angle)
        claw1_y = end_y + dir_y / norm * claw_length * np.cos(claw_angle) + dir_x / norm * claw_length * np.sin(
            claw_angle)
        claw2_x = end_x + dir_x / norm * claw_length * np.cos(-claw_angle) - dir_y / norm * claw_length * np.sin(
            -claw_angle)
        claw2_y = end_y + dir_y / norm * claw_length * np.cos(-claw_angle) + dir_x / norm * claw_length * np.sin(
            -claw_angle)
        robot_state['claw'].set_data([end_x, claw1_x, end_x, claw2_x], [end_y, claw1_y, end_y, claw2_y])

        robot_state['last_pos'] = (end_x, end_y)
        if target_name == 'LLA' and target_name in state_manager['station_wafers']:
            state_manager['station_wafers'][target_name].set_visible(False)
            stop_wafer_flash(state_manager['station_wafers'][target_name])


# ---------------------- 交互控制函数 ----------------------
def play_next_step(ax):
    """执行下一步动作（按钮/回车触发，微调状态提示）"""
    if state_manager['animation_running']:
        return

    # 检查是否完成所有任务
    state_manager['current_step'] += 1
    if state_manager['current_step'] >= state_manager['total_steps']:
        state_manager['status_label'].config(text="✅ 所有动作已执行完毕！PM工位晶圆将持续灰金闪烁至程序关闭")
        return

    # 更新进度和状态
    current, total = state_manager['current_step'] + 1, state_manager['total_steps']
    state_manager['status_label'].config(text=f"🔄 正在执行第 {current}/{total} 步动作... PM工位晶圆灰金循环闪烁")
    state_manager['progress_var'].set(current / total * 100)

    # 获取当前任务
    robot1_task, robot2_task = state_manager['task_queue'][state_manager['current_step']]
    state_manager['animation_running'] = True
    current_frame = [0]  # 帧计数器（列表实现可变变量）

    def animate_frame():
        """逐帧播放动画"""
        frame = current_frame[0]
        # 执行双机械臂动作
        execute_robot_action(1, robot1_task, frame, ax)
        execute_robot_action(2, robot2_task, frame, ax)
        # 保障LLA晶圆隐藏并停止闪烁
        if 'LLA' in state_manager['station_wafers']:
            state_manager['station_wafers']['LLA'].set_visible(False)
            stop_wafer_flash(state_manager['station_wafers']['LLA'])
        # 更新画布
        state_manager['canvas'].draw()

        current_frame[0] += 1
        # 动画结束判断
        if current_frame[0] > state_manager['frames_per_step']:
            state_manager['animation_running'] = False
            state_manager['step_animation'] = None
            state_manager['status_label'].config(
                text=f"✅ 第 {current}/{total} 步完成 | 按回车/点按钮执行下一步 | PM工位晶圆灰金循环闪烁")
            return
        # 继续播放下一帧
        state_manager['step_animation'] = state_manager['root'].after(50, animate_frame)

    # 启动动画
    animate_frame()


# ---------------------- 创建GUI界面 ----------------------
def create_full_gui(task_matrix):
    """创建带加工路径和PM工位灰金闪烁功能的交互界面"""
    # 验证任务矩阵
    for step_idx, (task1, task2) in enumerate(task_matrix):
        if task1 != 0 and (len(task1) != 2 or task1[0] not in STATIONS or task1[1] not in STATIONS):
            raise ValueError(f"第{step_idx}步：机械臂1任务格式错误")
        if task2 != 0 and (len(task2) != 2 or task2[0] not in STATIONS or task2[1] not in STATIONS):
            raise ValueError(f"第{step_idx}步：机械臂2任务格式错误")

    # 初始化任务状态
    state_manager['task_queue'] = task_matrix
    state_manager['total_steps'] = len(task_matrix)

    # 创建Tkinter主窗口
    root = tk.Tk()
    root.title("Processing System of Single-arm Multicluster Tools ")
    root.geometry("1400x950")  # 微调窗口高度，适配新按钮
    state_manager['root'] = root

    # 窗口关闭时，清理所有闪烁定时器和动画，避免报错
    def on_closing():
        stop_wafer_flash()  # 停止所有晶圆闪烁
        if state_manager['step_animation']:
            root.after_cancel(state_manager['step_animation'])
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 创建绘图区域（匹配参考图背景）
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#f8f8f8')  # 浅灰色背景，匹配参考图

    # 绘制静态元素
    # 1. 绘制所有工位（自动调用修改后的八边形绘制函数）
    for name, (x, y, color) in STATIONS.items():
        draw_station(ax, x, y, color, name)

    # 2. 绘制机械臂底座
    draw_robot_base(ax, 1)
    draw_robot_base(ax, 2)

    # 3. 初始化机械臂
    init_robot(ax, 1)
    init_robot(ax, 2)

    # 4. 绘制所有任务流向箭头（参考图样式）
    draw_all_task_arrows(ax, task_matrix)

    # 嵌入Matplotlib画布到Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    state_manager['canvas'] = canvas

    # 控制面板
    control_frame = tk.Frame(root, padx=20, pady=10, bg='#f0f0f0')
    control_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # 状态标签（新增灰金闪烁提示）
    status_label = tk.Label(
        control_frame,
        text=f"📋 准备就绪 | 共{state_manager['total_steps']}步 | PM工位晶圆灰金循环闪烁 | LL工位静态灰色 | 按回车/点按钮开始执行",
        font=("SimHei", 12), bg='#f0f0f0', padx=10, pady=5
    )
    status_label.pack(side=tk.TOP, anchor=tk.W)
    state_manager['status_label'] = status_label

    # 进度条
    progress_var = tk.DoubleVar()
    progress_bar = tk.Scale(
        control_frame, variable=progress_var, from_=0, to=100, orient=tk.HORIZONTAL,
        length=800, label="执行进度", font=("SimHei", 10), state=tk.DISABLED
    )
    progress_bar.pack(side=tk.TOP, pady=5)
    state_manager['progress_var'] = progress_var

    # 按钮框架
    button_frame = tk.Frame(control_frame, bg='#f0f0f0')
    button_frame.pack(side=tk.TOP, pady=10)

    # 执行下一步按钮
    next_btn = tk.Button(
        button_frame, text="▶️执行动作", command=lambda: play_next_step(ax),
        font=("SimHei", 12), width=12, height=2, bg='#4CAF50', fg='white'
    )
    next_btn.pack(side=tk.LEFT, padx=8)

    # 控制普通箭头显示/隐藏的按钮
    global toggle_btn
    toggle_btn = tk.Button(
        button_frame, text="🔴 机械臂箭头", command=toggle_arrows_visibility,
        font=("SimHei", 12), width=12, height=2, bg='#FF9800', fg='white'
    )
    toggle_btn.pack(side=tk.LEFT, padx=8)

    # 晶圆加工路径显示/隐藏按钮
    global process_btn
    process_btn = tk.Button(
        button_frame, text="🔵 加工路径", command=toggle_process_path,
        font=("SimHei", 12), width=12, height=2, bg='#2196F3', fg='white'
    )
    process_btn.pack(side=tk.LEFT, padx=8)

    # 退出程序按钮（绑定清理逻辑）
    exit_btn = tk.Button(
        button_frame, text="❌ 退出程序", command=on_closing,
        font=("SimHei", 12), width=12, height=2, bg='#555555', fg='white'
    )
    exit_btn.pack(side=tk.LEFT, padx=8)

    # 绑定回车键
    root.bind('<Return>', lambda event: play_next_step(ax))

    # 操作说明（新增灰金闪烁和工位规则说明）
    info_label = tk.Label(
        control_frame,
        text="📝 操作说明：按回车键/点击「执行下一步」按钮逐步执行 | PM1-PM10工位=晶圆闪烁（加工中） | LLC/LLD/LLA/LLB工位=晶圆静态（非加工）",
        font=("SimHei", 10), bg='#f0f0f0', fg='#666666', wraplength=1300
    )
    info_label.pack(side=tk.TOP, pady=5)

    # 启动主循环
    root.mainloop()


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 任务矩阵（保持原有逻辑不变）
    task_matrix = [
        (('LLA', 'PM1'), 0), (('LLA', 'PM10'), 0), (('PM1', 'PM2'), 0), (('LLA', 'PM1'), 0),
        (('PM10', 'PM9'), 0), (('LLA', 'PM10'), 0), (('PM2', 'LLC'), 0), (('PM1', 'PM2'), ('LLC', 'PM3')),
        (('PM9', 'LLC'), 0), (('PM10', 'PM9'), ('LLC', 'PM8')), (('LLA', 'PM1'), 0), (('LLA', 'PM10'), 0),
        (('PM2', 'LLC'), ('PM3', 'PM4')), (0, ('LLC', 'PM3')), (('PM9', 'LLC'), ('PM8', 'PM7')),
        (('PM1', 'PM2'), ('LLC', 'PM8')),
        (('PM10', 'PM9'), 0), (('LLA', 'PM1'), ('PM4', 'PM5')), (('LLA', 'PM10'), 0), (0, ('PM3', 'PM4')),
        (0, ('PM7', 'PM6')), (0, ('PM8', 'PM7')), (('PM2', 'LLC'), 0), (0, ('LLC', 'PM3')),
        (('PM9', 'LLC'), 0), (('PM1', 'PM2'), ('PM5', 'LLD')), (('PM10', 'PM9'), ('LLC', 'PM8')),
        (('LLD', 'LLA'), ('PM4', 'PM5')),
        (('LLA', 'PM1'), ('PM6', 'LLD')), (('LLA', 'PM10'), ('PM7', 'PM6')), (('LLD', 'LLA'), 0), (0, ('PM3', 'PM4')),
        (('PM2', 'LLC'), ('PM8', 'PM7')), (0, ('LLC', 'PM3')), (('PM9', 'LLC'), 0), (0, ('PM5', 'LLD')),
        (('LLD', 'LLA'), 0), (('PM1', 'PM2'), ('PM6', 'LLD')), (('PM10', 'PM9'), 0), (('LLD', 'LLA'), ('PM4', 'PM5')),
        (('LLA', 'PM1'), ('LLC', 'PM8')), (('LLA', 'PM10'), ('PM7', 'PM6')), (0, ('PM3', 'PM4')),
        (('PM2', 'LLC'), ('PM5', 'LLD')),
        (0, ('LLC', 'PM3')), (('PM9', 'LLC'), 0), (('PM1', 'PM2'), ('LLC', 'PM8')), (('PM10', 'PM9'), ('PM4', 'PM5')),
        (('LLA', 'PM1'), ('PM6', 'LLD')), (('LLD', 'LLA'), 0), (('LLA', 'PM10'), 0), (0, ('PM7', 'PM6')),
        (0, ('PM3', 'PM4')), (0, ('PM8', 'PM7')), (('PM2', 'LLC'), ('PM5', 'LLD')), (('LLD', 'LLA'), 0),
        (0, ('LLC', 'PM3')), (('PM9', 'LLC'), 0), (('PM1', 'PM2'), ('LLC', 'PM8')), (('PM10', 'PM9'), ('PM6', 'LLD')),
        (('LLD', 'LLA'), ('PM4', 'PM5')), (('LLA', 'PM1'), ('PM7', 'PM6')), (('LLA', 'PM10'), ('PM3', 'PM4')),
        (0, ('PM8', 'PM7')),
        (('PM2', 'LLC'), ('PM5', 'LLD')), (0, ('LLC', 'PM3')), (('PM9', 'LLC'), 0), (('LLD', 'LLA'), 0),
        (0, ('PM6', 'LLD')), (('PM1', 'PM2'), 0), (('PM10', 'PM9'), ('PM4', 'PM5')), (('LLD', 'LLA'), ('LLC', 'PM8')),
        (('LLA', 'PM1'), ('PM7', 'PM6')), (('LLA', 'PM10'), ('PM3', 'PM4')), (('PM2', 'LLC'), ('PM5', 'LLD')),
        (('LLD', 'LLA'), ('LLC', 'PM3')),
        (('PM9', 'LLC'), ('PM8', 'PM7')), (('PM1', 'PM2'), ('PM6', 'LLD')), (('LLD', 'LLA'), ('PM4', 'PM5')),
        (('PM10', 'PM9'), 0),
        (0, ('LLC', 'PM8')), (0, ('PM3', 'PM4')), (0, ('PM7', 'PM6')), (('PM2', 'LLC'), ('PM5', 'LLD')),
        (('LLD', 'LLA'), ('LLC', 'PM3')), (('PM9', 'LLC'), ('PM8', 'PM7')), (0, ('LLC', 'PM8')), (0, ('PM4', 'PM5')),
        (0, ('PM6', 'LLD')), (('LLD', 'LLA'), 0), (0, ('PM3', 'PM4')), (0, ('PM7', 'PM6')),
        (0, ('PM8', 'PM7')), (0, ('PM5', 'LLD')), (('LLD', 'LLA'), 0), (0, ('PM4', 'PM5')),
        (0, ('PM6', 'LLD')), (('LLD', 'LLA'), 0), (0, ('PM7', 'PM6')), (0, ('PM5', 'LLD')),
        (('LLD', 'LLA'), 0), (0, ('PM6', 'LLD')), (('LLD', 'LLA'), 0)
    ]
    # 启动交互界面
    create_full_gui(task_matrix)