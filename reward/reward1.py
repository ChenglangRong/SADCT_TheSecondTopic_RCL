import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def main(file_info, window_size=80, save_path='./moving_average_rewards.png'):
    """
    读取多智能体算法的Excel奖励数据，计算移动平均并生成对比图

    参数说明:
    - file_info: 字典，键为Excel文件路径，值为对应的算法名称
    - window_size: 移动平均窗口大小，默认100
    - save_path: 图像保存路径，默认当前目录
    """
    # 1. 基础配置：设置中文字体（解决中文显示乱码问题）
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 2. 移动平均计算函数
    def calculate_moving_average(rewards, window_size):
        """
        计算指定窗口大小的移动平均奖励
        :param rewards: 原始奖励数组（需去除NaN值）
        :param window_size: 移动平均窗口大小
        :return: 移动平均后的奖励数组
        """
        if len(rewards) < window_size:
            raise ValueError(f"原始数据长度({len(rewards)})小于窗口大小({window_size})，无法计算移动平均")
        # 使用numpy卷积实现移动平均，mode='valid'确保输出长度为len(rewards)-window_size+1
        return np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    # 3. 数据读取与预处理
    algorithm_data = {}  # 存储各算法的处理后数据
    for file_path, algo_name in file_info.items():
        try:
            # 读取Excel文件（默认读取第一个工作表）
            df = pd.read_excel(file_path)

            # 检查是否存在'reward'列
            if 'reward' not in df.columns:
                print(f"⚠️  警告：{algo_name}的文件中未找到'reward'列，已跳过该算法")
                continue

            # 提取有效奖励数据（去除NaN值）
            valid_rewards = df['reward'].dropna().values
            if len(valid_rewards) == 0:
                print(f"⚠️  警告：{algo_name}没有有效奖励数据，已跳过该算法")
                continue

            # 计算移动平均
            moving_avg_rewards = calculate_moving_average(valid_rewards, window_size)

            # 保存数据（步数从0开始计数）
            algorithm_data[algo_name] = {
                'steps': np.arange(len(moving_avg_rewards)),  # 步数序列
                'moving_avg_rewards': moving_avg_rewards  # 移动平均奖励
            }

            # 打印处理结果日志
            print(f"✅ 成功处理{algo_name}：")
            print(f"   - 原始数据长度：{len(valid_rewards)}")
            print(f"   - 移动平均后长度：{len(moving_avg_rewards)}")
            print(f"   - 奖励范围：{moving_avg_rewards.min():.2f} ~ {moving_avg_rewards.max():.2f}\n")

        except Exception as e:
            print(f"❌ 处理{algo_name}时出错：{str(e)}\n")

    # 4. 生成可视化图表
    if not algorithm_data:
        print("❌ 没有有效数据可生成图表")
        return

    # 创建画布（设置合适的尺寸）
    plt.figure(figsize=(14, 9))

    # 定义颜色和线型（确保6种算法有明显区分）
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '-', '-', '-', '-', '-']

    # 绘制各算法的移动平均曲线
    for idx, (algo_name, data) in enumerate(algorithm_data.items()):
        plt.plot(
            data['steps'],  # x轴：步数
            data['moving_avg_rewards'],  # y轴：移动平均奖励
            color=color_palette[idx % len(color_palette)],  # 循环使用颜色
            linestyle=line_styles[idx % len(line_styles)],  # 循环使用线型
            linewidth=2.5,  # 线条宽度（确保清晰可见）
            label=algo_name  # 图例标签（算法名称）
        )

    # -------------------------- 核心修改：XY轴刻度美化 --------------------------
    plt.xlabel('Episode', fontsize=23, fontweight='bold')  # x轴标签
    plt.ylabel(f'Moving cumulated episode reward', fontsize=23, fontweight='bold')  # y轴标签
    plt.title('Training Convergence Curve Comparison of Reinforcement Learning Algorithms',
              fontsize=27, fontweight='bold', pad=20)  # 标题
    plt.grid(True, alpha=0.7, linestyle='-', linewidth=0.5)  # 网格线
    plt.legend(loc='best', fontsize=20, framealpha=0.9)  # 图例
    plt.xlim(left=0)  # x轴从0开始

    # 设置X轴刻度：字体20号+粗体+深黑色（变⼤+变深核心）
    plt.xticks(fontsize=20, fontweight='bold', color='#000000')
    # 设置Y轴刻度：字体20号+粗体+深黑色（与X轴保持一致，视觉统一）
    plt.yticks(fontsize=20, fontweight='bold', color='#000000')

    plt.tight_layout()  # 自动调整布局
    # 保存图表（高分辨率300DPI，bbox_inches='tight'防止刻度截断）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布（释放内存）

    print(f"📊 图表已保存至：{save_path}")


# 5. 主程序入口（直接运行时执行）
if __name__ == "__main__":
    # --------------------------
    # 配置参数（可根据需求修改）
    # --------------------------
    # 1. 文件路径与算法名称映射（需根据实际文件位置调整）
    FILE_CONFIG = {
        'C:\\Users\\Lenovo\\Desktop\\data1\\MAPPO_data.xlsx': 'MAPPO',
        'C:\\Users\\Lenovo\\Desktop\\data1\\MAQMix_data.xlsx': 'MAQMIX',
        'C:\\Users\\Lenovo\\Desktop\\data1\\MultiDDPG_data.xlsx': 'MADDPG',
        'C:\\Users\\Lenovo\\Desktop\\data1\\MultiMACTD4_data.xlsx': 'MACTD4',
        'C:\\Users\\Lenovo\\Desktop\\data1\\MultiMATD3_data.xlsx': 'MATD3',
        'C:\\Users\\Lenovo\\Desktop\\data1\\QmixPPOAC_data.xlsx': 'MAQmix-PPO'
    }

    # 2. 移动平均窗口大小（当前为250，可按需调整）
    MOVING_AVG_WINDOW = 250

    # 3. 图表保存路径（默认保存在当前目录）
    OUTPUT_IMAGE_PATH = 'multi_agent_reward_comparison1.png'

    # 执行主函数
    main(FILE_CONFIG, MOVING_AVG_WINDOW, OUTPUT_IMAGE_PATH)