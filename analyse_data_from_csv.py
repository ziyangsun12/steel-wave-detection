#!/usr/bin/env python3
"""
从CSV文件读取带钢浪形检测数据并分析浪形出现的时间戳
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def read_detection_data(csv_path='output/detection_results.csv'):
    """读取检测数据

    Args:
        csv_path: CSV文件路径

    Returns:
        DataFrame: 检测数据
    """
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在 - {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条检测记录")
        return df
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None


def convert_to_beijing_time(df):
    """将时间戳转换为北京时间(UTC+8)

    Args:
        df: 包含timestamp列的DataFrame

    Returns:
        DataFrame: 添加了datetime和beijing_time列的数据
    """
    # 将Unix时间戳转换为UTC时间
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

    # 转换为北京时间 (UTC+8)
    df['beijing_time'] = df['datetime_utc'].dt.tz_convert('Asia/Shanghai')

    # 为了兼容性，保留原来的datetime列（也设置为北京时间）
    df['datetime'] = df['beijing_time']

    return df


def analyze_wave_timestamps(df):
    """分析浪形出现的时间戳

    Args:
        df: 检测数据DataFrame

    Returns:
        dict: 分析结果
    """
    if df is None or len(df) == 0:
        print("没有数据可分析")
        return {}

    # 将时间戳转换为北京时间(UTC+8)
    df = convert_to_beijing_time(df)

    # 筛选出有浪形的数据（排除"未知"和"平直"）
    wave_types_to_analyze = ['中浪', '双边浪', 'DS单边浪', 'WS单边浪', '严重']
    wave_data = df[df['wave_type'].isin(wave_types_to_analyze)].copy()

    # 或者筛选出浪高大于阈值的数据
    wave_height_threshold = 0.5  # 浪高阈值(mm)
    high_wave_data = df[df['wave_height'] > wave_height_threshold].copy()

    analysis_result = {
        'total_frames': len(df),
        'wave_frames': len(wave_data),
        'high_wave_frames': len(high_wave_data),
        'wave_ratio': len(wave_data) / len(df) * 100 if len(df) > 0 else 0,
    }

    print("\n" + "=" * 60)
    print("带钢浪形检测结果统计分析")
    print("=" * 60)
    print(f"\n总检测帧数: {analysis_result['total_frames']}")
    print(f"检测到浪形帧数: {analysis_result['wave_frames']}")
    print(f"浪高>{wave_height_threshold}mm的帧数: {analysis_result['high_wave_frames']}")
    print(f"浪形出现比例: {analysis_result['wave_ratio']:.2f}%")

    # 按浪形类型统计
    if len(wave_data) > 0:
        print("\n" + "-" * 60)
        print("浪形类型分布:")
        print("-" * 60)
        type_counts = wave_data['wave_type'].value_counts()
        for wave_type, count in type_counts.items():
            percentage = count / len(wave_data) * 100
            print(f"  {wave_type}: {count} 帧 ({percentage:.2f}%)")

        # 按浪形等级统计
        print("\n" + "-" * 60)
        print("浪形等级分布:")
        print("-" * 60)
        level_counts = wave_data['wave_level'].value_counts()
        for level, count in level_counts.items():
            percentage = count / len(wave_data) * 100
            print(f"  {level}: {count} 帧 ({percentage:.2f}%)")

        # 浪高统计
        print("\n" + "-" * 60)
        print("浪高统计:")
        print("-" * 60)
        print(f"  平均浪高: {wave_data['wave_height'].mean():.2f} mm")
        print(f"  最大浪高: {wave_data['wave_height'].max():.2f} mm")
        print(f"  最小浪高: {wave_data['wave_height'].min():.2f} mm")
        print(f"  浪高标准差: {wave_data['wave_height'].std():.2f} mm")

        # 浪宽统计
        print("\n" + "-" * 60)
        print("浪宽统计:")
        print("-" * 60)
        valid_width = wave_data[wave_data['wave_width'] > 0]
        if len(valid_width) > 0:
            print(f"  平均浪宽: {valid_width['wave_width'].mean():.2f} mm")
            print(f"  最大浪宽: {valid_width['wave_width'].max():.2f} mm")
            print(f"  最小浪宽: {valid_width['wave_width'].min():.2f} mm")

        # 浪形出现的时间段分析
        print("\n" + "-" * 60)
        print("浪形出现的时间段:")
        print("-" * 60)

        if len(wave_data) > 0:
            first_wave_time = wave_data['beijing_time'].min()
            last_wave_time = wave_data['beijing_time'].max()
            print(f"  首次检测到浪形: {first_wave_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            print(f"  最后检测到浪形: {last_wave_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            print(f"  浪形持续时间: {(last_wave_time - first_wave_time).total_seconds():.2f} 秒")

            # 找出连续浪形时间段
            print("\n  浪形出现的具体时间戳:")
            print("  " + "-" * 56)
            print(f"  {'序号':<6} {'时间戳':<28} {'浪形类型':<12} {'浪高(mm)':<10}")
            print("  " + "-" * 56)

            for idx, row in wave_data.iterrows():
                time_str = row['beijing_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"  {idx + 1:<6} {time_str:<28} {row['wave_type']:<12} {row['wave_height']:<10.2f}")

    else:
        print("\n未检测到明显的浪形缺陷")
        print("提示: 当前数据中所有记录的浪形类型为'未知'或'平直'")
        print("      可能需要检查检测算法或视频源")

    # 时间序列分析
    print("\n" + "-" * 60)
    print("时间序列分析:")
    print("-" * 60)

    if len(df) > 1:
        time_range = (df['timestamp'].max() - df['timestamp'].min())
        print(f"  检测时间范围: {time_range:.2f} 秒")
        print(f"  平均检测频率: {len(df) / time_range:.2f} FPS" if time_range > 0 else "  无法计算FPS")

        # 检测开始和结束时间（北京时间）
        start_time = df['beijing_time'].min()
        end_time = df['beijing_time'].max()
        print(f"  检测开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  检测结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return analysis_result


def export_wave_events(df, output_path='output/wave_events.csv'):
    """导出浪形事件到CSV文件

    Args:
        df: 检测数据DataFrame
        output_path: 输出文件路径
    """
    if df is None or len(df) == 0:
        print("没有数据可导出")
        return

    # 确保有时间转换
    if 'beijing_time' not in df.columns:
        df = convert_to_beijing_time(df)

    # 筛选出有浪形的数据
    wave_types_to_export = ['中浪', '双边浪', 'DS单边浪', 'WS单边浪']
    wave_data = df[df['wave_type'].isin(wave_types_to_export)].copy()

    if len(wave_data) == 0:
        print("未检测到浪形事件，无需导出")
        return

    # 添加可读时间列（北京时间）
    wave_data['readable_time'] = wave_data['beijing_time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # 选择需要的列
    export_columns = [
        'timestamp', 'readable_time', 'wave_type', 'wave_level',
        'wave_height', 'wave_width', 'wave_position_x', 'wave_position_y'
    ]

    wave_data[export_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n浪形事件已导出到: {output_path}")
    print(f"共导出 {len(wave_data)} 条浪形记录")


def generate_summary_report(df, report_path='output/wave_analysis_report.txt'):
    """生成分析报告

    Args:
        df: 检测数据DataFrame
        report_path: 报告文件路径
    """
    if df is None or len(df) == 0:
        print("没有数据可生成报告")
        return

    # 确保有时间转换
    if 'beijing_time' not in df.columns:
        df = convert_to_beijing_time(df)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("带钢浪形检测报告\n")
        f.write("=" * 70 + "\n\n")

        # 基本信息
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)\n")
        f.write(f"数据文件: output/detection_results.csv\n")
        f.write(f"总检测帧数: {len(df)}\n\n")

        # 时间信息（北京时间）
        f.write(f"检测开始时间: {df['beijing_time'].min().strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)\n")
        f.write(f"检测结束时间: {df['beijing_time'].max().strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)\n")
        f.write(f"检测时长: {(df['beijing_time'].max() - df['beijing_time'].min()).total_seconds():.2f} 秒\n\n")

        # 浪形统计
        wave_types = ['中浪', '双边浪', 'DS单边浪', 'WS单边浪']
        wave_data = df[df['wave_type'].isin(wave_types)]

        f.write("-" * 70 + "\n")
        f.write("浪形检测结果汇总\n")
        f.write("-" * 70 + "\n\n")

        if len(wave_data) > 0:
            f.write(f"检测到浪形帧数: {len(wave_data)}\n")
            f.write(f"浪形出现率: {len(wave_data) / len(df) * 100:.2f}%\n\n")

            f.write("浪形类型分布:\n")
            for wave_type in wave_types:
                count = len(wave_data[wave_data['wave_type'] == wave_type])
                if count > 0:
                    f.write(f"  - {wave_type}: {count} 帧\n")

            f.write("\n浪高等级分布:\n")
            for level in ['低', '中', '高', '严重']:
                count = len(wave_data[wave_data['wave_level'] == level])
                if count > 0:
                    f.write(f"  - {level}: {count} 帧\n")

            f.write(f"\n浪高统计:\n")
            f.write(f"  - 平均值: {wave_data['wave_height'].mean():.2f} mm\n")
            f.write(f"  - 最大值: {wave_data['wave_height'].max():.2f} mm\n")
            f.write(f"  - 最小值: {wave_data['wave_height'].min():.2f} mm\n")

            f.write(f"\n浪形位置统计:\n")
            f.write(
                f"  - X坐标范围: {wave_data['wave_position_x'].min():.2f} ~ {wave_data['wave_position_x'].max():.2f} mm\n")
            f.write(
                f"  - Y坐标范围: {wave_data['wave_position_y'].min():.2f} ~ {wave_data['wave_position_y'].max():.2f} mm\n")
        else:
            f.write("未检测到明显浪形缺陷\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"\n分析报告已生成: {report_path}")


def main():
    """主函数"""
    print("带钢浪形检测数据分析工具")
    print("=" * 60)

    # 读取数据
    csv_path = 'output/detection_results.csv'
    df = read_detection_data(csv_path)

    if df is None:
        return

    # 分析浪形时间戳
    analysis_result = analyze_wave_timestamps(df)

    # 导出浪形事件
    export_wave_events(df)

    # 生成分析报告
    generate_summary_report(df)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
