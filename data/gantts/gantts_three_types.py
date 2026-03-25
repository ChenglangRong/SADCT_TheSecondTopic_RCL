# 导入所需库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches  # 绘制矩形边框

# 设置matplotlib默认字体为Times New Roman（解决中文+英文混排兼容问题）
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def generate_combined_wafer_gantt(pm_excel_path, ll_excel_path, save_path='three_type_wafer_gantt_chart.png'):
    """
    生成整合版晶圆加工甘特图
    适配新格式：
    - PM文件单元格为"开始时间 驻留时长"
    - BM文件单元格为"开始时间 结束时间"（展示晶圆存在时长）
    调整：
    - LL取出时间左上角+放入时间左下角+X轴限定2350+自定义色系
    - BM标签：左侧显示开始时间，上方显示持续时长，不显示结束时间
    - 每个模块行添加黑色边框，划分空白区域
    - BM模块新增与PM一致的红+蓝前置块、绿色后置块（无驻留红块）
    - 新增：红色驻留块上方显示驻留时长数字
    """

    # --------------------------
    # 1. 基础设置与参数定义
    # --------------------------
    processing_times = {
        "PM11": 85, "PM12": 99, "PM15": 94, "PM16": 80,
        "BM1": 0, "BM2": 0,  # BM时长从Excel中解析
        "PM21": 101, "PM22": 120, "PM23": 120, "PM24": 120, "PM25": 120, "PM26": 115,
        "取出晶圆": 10, "放入晶圆": 5
    }

    # 颜色配置（保持最新自定义色系）
    colors = {
        "PM11": '#FFA751', "PM12": '#FFA751', "PM15": '#FFA751', "PM16": '#FFA751',
        "PM21": '#FFA751', "PM22": '#FFA751', "PM23": '#FFA751', "PM24": '#FFA751', "PM25": '#FFA751',
        "PM26": '#FFA751',
        "BM1": '#FFA751', "BM2": '#FFA751',
        "取出晶圆": '#006400', "放入晶圆": '#87CEFA'
    }

    # 前置/后置块配置（保持不变，BM与PM共用）
    block_config = {
        'red_prefix': {'duration': 2, 'color': '#00008B', 'alpha': 0.7},
        'blue_prefix': {'duration': 5, 'color': '#87CEFA', 'alpha': 0.7},
        'residence_red': {'color': '#CC0000', 'alpha': 0.8},
        'blue_suffix': {'duration': 5, 'color': '#006400', 'alpha': 0.7}
    }

    # 仅LL跳过前置/后置块，BM1/BM2移除跳过
    skip_block_modules = ["LL"]
    ll_label_fontsize = 9  # LL字体大小
    x_axis_max = 2350  # X轴强制最大值2350
    bm_duration_fontsize = 8  # BM持续时长字体大小
    bm_start_fontsize = 9  # BM开始时间字体大小
    residence_label_fontsize = 8  # 新增：驻留时长标签字体大小

    # 模块边框配置
    border_config = {
        'linewidth': 1,  # 边框线宽
        'color': 'black',  # 边框颜色
        'linestyle': '-',  # 边框线型
        'alpha': 1.0,  # 边框透明度
        'padding': 0.15  # 边框上下内边距
    }

    # --------------------------
    # 2. 数据读取与预处理（无修改）
    # --------------------------
    pm_df = pd.read_excel(pm_excel_path, dtype=str)
    ll_df = pd.read_excel(ll_excel_path)
    ll_df = ll_df[["取出晶圆", "放入晶圆"]].copy()

    all_gantt_data = []
    # 处理PM/BM模块
    for module in pm_df.columns:
        if module in processing_times:
            cell_values = pm_df[module].dropna().values
            for i, cell_val in enumerate(cell_values):
                try:
                    # 区分BM和PM的解析逻辑
                    if module in ["BM1", "BM2"]:
                        # BM格式：开始时间 结束时间
                        start_time_str, end_time_str = str(cell_val).strip().split(' ')
                        start_time = float(start_time_str)
                        end_time = float(end_time_str)
                        process_time = end_time - start_time  # BM的时长=结束时间-开始时间
                        residence_time = 0  # BM无驻留时长概念
                    else:
                        # PM格式：开始时间 驻留时长
                        start_time_str, residence_time_str = str(cell_val).strip().split(' ')
                        start_time = float(start_time_str)
                        residence_time = float(residence_time_str)
                        process_time = processing_times[module]
                        end_time = start_time + process_time
                except (ValueError, IndexError):
                    # 异常处理：无法解析时默认值
                    if module in ["BM1", "BM2"]:
                        start_time = float(cell_val) if str(cell_val).replace('.', '').isdigit() else 0
                        end_time = start_time
                        process_time = 0
                        residence_time = 0
                    else:
                        start_time = float(cell_val) if str(cell_val).replace('.', '').isdigit() else 0
                        residence_time = 0
                        process_time = processing_times[module]
                        end_time = start_time + process_time
                    print(f"警告：{module}第{i + 1}行格式异常({cell_val})，使用默认值")

                all_gantt_data.append({
                    'module': module, 'task_id': f"{module}_{i + 1}",
                    'start_time': start_time, 'end_time': end_time,
                    'process_time': process_time, 'residence_time': residence_time,
                    'original_module': module, 'task_index': i
                })

    # 处理LL模块（保持不变）
    ll_operations = ["取出晶圆", "放入晶圆"]
    for operation in ll_operations:
        start_times = ll_df[operation].dropna().values
        process_time = processing_times[operation]
        for i, start_time in enumerate(start_times):
            end_time = start_time + process_time
            all_gantt_data.append({
                'module': "LL", 'task_id': f"LL_{operation}_{i + 1}",
                'start_time': start_time, 'end_time': end_time,
                'process_time': process_time, 'original_module': operation,
                'residence_time': 0, 'task_index': i
            })
    gantt_df = pd.DataFrame(all_gantt_data)

    # --------------------------
    # 3. 绘制整合版甘特图
    # --------------------------
    fig, ax = plt.subplots(figsize=(40, 10))
    module_display_order = [
        "PM11", "PM12", "PM15", "PM16", "BM1", "BM2",
        "PM21", "PM22", "PM23", "PM24", "PM25", "PM26", "LL"
    ]
    available_modules = [m for m in module_display_order if m in gantt_df['module'].unique()]
    y_pos = range(len(available_modules))
    bar_height = 0.3

    # 绘制每个模块行的黑色边框（底层）
    for i, module in enumerate(available_modules):
        y_min = i - border_config['padding']
        y_max = i + border_config['padding']
        min_time = gantt_df['start_time'].min() - (
                block_config['red_prefix']['duration'] + block_config['blue_prefix']['duration'] + 10)
        x_min = min_time if min_time != 0 else -20
        x_max = x_axis_max

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=border_config['linewidth'],
            edgecolor=border_config['color'],
            linestyle=border_config['linestyle'],
            facecolor='none',
            alpha=border_config['alpha'],
            zorder=0
        )
        ax.add_patch(rect)

    # 绘制甘特条（BM与PM共用前置/后置块逻辑，仅BM无驻留红块）
    for i, module in enumerate(available_modules):
        module_data = gantt_df[gantt_df['module'] == module].sort_values('start_time')
        for _, row in module_data.iterrows():
            duration = row['end_time'] - row['start_time']
            residence_time = row['residence_time']

            # 绘制前置/驻留/后置块（仅LL跳过，BM/PM都绘制，BM无驻留红块）
            if module not in skip_block_modules:
                # 计算前置块起始位置（红+蓝，与PM完全一致）
                red_p_start = row['start_time'] - (
                        block_config['red_prefix']['duration'] + block_config['blue_prefix']['duration'])
                red_p_end = red_p_start + block_config['red_prefix']['duration']
                blue_p_start = red_p_end
                blue_p_end = blue_p_start + block_config['blue_prefix']['duration']

                # 绘制红色前置块
                ax.barh(i, block_config['red_prefix']['duration'],
                        left=red_p_start, height=bar_height,
                        color=block_config['red_prefix']['color'], alpha=block_config['red_prefix']['alpha'],
                        edgecolor='white', linewidth=0.6, zorder=2)
                # 绘制蓝色前置块
                ax.barh(i, block_config['blue_prefix']['duration'],
                        left=blue_p_start, height=bar_height,
                        color=block_config['blue_prefix']['color'], alpha=block_config['blue_prefix']['alpha'],
                        edgecolor='white', linewidth=0.6, zorder=2)

                # 驻留红块：仅PM绘制（BM residence_time恒为0，自动跳过）
                if residence_time > 0:
                    res_start = row['end_time']
                    res_end = res_start + residence_time
                    ax.barh(i, residence_time,
                            left=res_start, height=bar_height,
                            color=block_config['residence_red']['color'], alpha=block_config['residence_red']['alpha'],
                            edgecolor='white', linewidth=0.6, zorder=2)

                    # 新增：在红色驻留块上方显示驻留时长数字
                    res_center_x = res_start + residence_time / 2  # 驻留块水平中心位置
                    res_label_y = i + bar_height / 2 + 0.05  # 驻留块上方位置
                    ax.text(res_center_x,
                            res_label_y,
                            f"{int(residence_time)}",
                            ha='center', va='bottom', fontsize=residence_label_fontsize, fontweight='bold',
                            color='black', fontfamily='Times New Roman', zorder=5)  # zorder确保文字在最上层

                # 绘制绿色后置块（BM与PM完全一致，无差异）
                blue_s_start = row['end_time'] + residence_time
                blue_s_end = blue_s_start + block_config['blue_suffix']['duration']
                ax.barh(i, block_config['blue_suffix']['duration'],
                        left=blue_s_start, height=bar_height,
                        color=block_config['blue_suffix']['color'], alpha=block_config['blue_suffix']['alpha'],
                        edgecolor='white', linewidth=0.6, zorder=2)

            # 绘制主方块（无修改，BM/PM/LL原有逻辑）
            if module == "LL":
                bar_color = colors[row['original_module']]
                alpha = 0.9
            else:
                bar_color = colors[module]
                alpha = 0.8

            ax.barh(i, duration, left=row['start_time'],
                    height=bar_height, color=bar_color, alpha=alpha,
                    edgecolor='white', linewidth=0.6, zorder=3)

            # 时间标签（无修改，BM/PM/LL原有规则）
            if module == "LL":
                ll_oper = row['original_module']
                if ll_oper == "取出晶圆":
                    ax.text(row['start_time'] + 6,
                            i + bar_height / 2 + 0.1,
                            f"{int(row['start_time'])}",
                            ha='right', va='bottom', fontsize=ll_label_fontsize, fontweight='bold',
                            color='black', fontfamily='Times New Roman', zorder=4)
                elif ll_oper == "放入晶圆":
                    ax.text(row['start_time'] - 0.5,
                            i + bar_height / 2 - 0.1,
                            f"{int(row['start_time'])}",
                            ha='right', va='top', fontsize=ll_label_fontsize, fontweight='bold',
                            color='black', fontfamily='Times New Roman', zorder=4)
            elif module in ["BM1", "BM2"]:
                # BM标签：左侧显示开始时间，上方显示持续时长，不显示结束时间
                ax.text(row['start_time'] - 10,
                        i + bar_height - 1 / 2 - 0.03,
                        f"{int(row['start_time'])}",
                        ha='left', va='top', fontsize=bm_start_fontsize, fontweight='bold',
                        color='black', fontfamily='Times New Roman', zorder=4)
                duration_center = row['start_time'] + duration / 2
                ax.text(duration_center,
                        i + bar_height / 2,
                        f"{int(duration)}",
                        ha='center', va='bottom', fontsize=bm_duration_fontsize, fontweight='bold',
                        color='black', fontfamily='Times New Roman', zorder=4)
            else:
                # PM标签：显示开始和结束时间
                ax.text(row['start_time'],
                        i + bar_height / 2 - 0.03,
                        f"{int(row['start_time'])}",
                        ha='left', va='top', fontsize=12, fontweight='bold',
                        color='black', fontfamily='Times New Roman', zorder=4)
                ax.text(row['end_time'],
                        i + bar_height / 2 - 0.03,
                        f"{int(row['end_time'])}",
                        ha='right', va='top', fontsize=12, fontweight='bold',
                        color='black', fontfamily='Times New Roman', zorder=4)

    # --------------------------
    # 4. 图表美化与设置（无修改）
    # --------------------------
    # Y轴设置
    ax.set_yticks(y_pos)
    ax.set_yticklabels(available_modules, fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylim(-0.4, len(available_modules) - 0.6)

    # X轴设置：最小值自适应，最大值强制为2350
    min_time = gantt_df['start_time'].min() - (
            block_config['red_prefix']['duration'] + block_config['blue_prefix']['duration'] + 10)
    ax.set_xlim(min_time, x_axis_max)

    # X轴样式
    ax.set_xlabel('Time(s)', fontsize=10, fontweight='bold', labelpad=15, fontfamily='Times New Roman')
    ax.tick_params(axis='x',
                   which='major',
                   length=8, width=2, color='black',
                   labelsize=10, labelcolor='black',
                   labelrotation=0, pad=8)
    ax.tick_params(axis='x', which='minor', length=4, width=1, color='black')

    # 标题与边框
    ax.set_title('three_type_wafer_gantt_chart',
                 fontsize=10, fontweight='bold', pad=25, fontfamily='Times New Roman')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_color('black')

    # 网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # 输出信息
    print(f"整合版甘特图已成功生成并保存为: {save_path}")


if __name__ == "__main__":
    pm_excel_path = "C:\\Users\\Lenovo\\Desktop\\SADCT-AIpha different\\photo\\gantts\\three_PMs.xlsx"
    ll_excel_path = "C:\\Users\\Lenovo\\Desktop\\SADCT-AIpha different\\photo\\gantts\\three_LL.xlsx"
    output_image_path = "C:\\Users\\Lenovo\\Desktop\\SADCT-AIpha different\\photo\\gantts\\three_type_wafer_gantt_chart.png"
    generate_combined_wafer_gantt(pm_excel_path, ll_excel_path, output_image_path)