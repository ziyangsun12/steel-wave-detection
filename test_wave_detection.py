#!/usr/bin/env python3
"""
带钢浪形检测测试脚本
测试data目录下所有视频的浪形识别效果
"""

import os
import sys
import json
import logging
from src.utils import Utils
from src.pipeline import Pipeline


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/test_wave_detection.log')
        ]
    )
    return logging.getLogger(__name__)


def extract_label_from_filename(filename):
    """从文件名提取浪形标签
    
    Args:
        filename: 文件名
    
    Returns:
        浪形标签
    """
    # 移除文件扩展名
    name = os.path.splitext(filename)[0]
    
    # 定义标签映射
    label_mappings = {
        '工作侧有单边浪': 'DS单边浪',
        '传动侧有单边浪': 'WS单边浪',
        '中浪': '中浪',
        '双边浪': '双边浪',
        '平直': '平直'
    }
    
    # 匹配标签
    for key, value in label_mappings.items():
        if key in name:
            return value
    
    return '未知'


def test_video(video_path, config):
    """测试单个视频
    
    Args:
        video_path: 视频路径
        config: 配置参数
    
    Returns:
        检测结果
    """
    # 覆盖配置
    config['camera']['type'] = 'local'
    config['camera']['video_path'] = video_path
    
    # 初始化Pipeline
    pipeline = Pipeline(config)
    
    try:
        # 启动Pipeline
        pipeline.start()
        
        # 等待视频处理完成
        import time
        while pipeline.running:
            time.sleep(1)
        
        # 获取检测结果
        results = pipeline.get_results()
        
        # 分析结果
        if results:
            # 统计最常见的浪形类型
            wave_types = [r.get('wave_type') for r in results if r.get('wave_type')]
            if wave_types:
                # 计算每种浪形的出现次数
                type_counts = {}
                for wave_type in wave_types:
                    type_counts[wave_type] = type_counts.get(wave_type, 0) + 1
                
                # 找出最常见的浪形
                dominant_type = max(type_counts, key=type_counts.get)
                return dominant_type
        
        return '未检测到'
    finally:
        # 停止Pipeline
        pipeline.stop()


def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    
    # 加载配置文件
    config_path = 'config/config.yaml'
    config = Utils.load_config(config_path)
    if not config:
        logger.error('加载配置文件失败，退出程序')
        sys.exit(1)
    
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 遍历data目录下所有视频文件
    data_folder = 'data'
    video_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                video_files.append(video_path)
    
    if not video_files:
        logger.error('data目录中未找到视频文件')
        sys.exit(1)
    
    # 限制测试视频数量，避免内存溢出
    max_videos = 5
    video_files = video_files[:max_videos]
    
    # 测试结果
    test_results = []
    correct_count = 0
    total_count = 0
    
    logger.info(f'开始测试 {len(video_files)} 个视频文件...')
    
    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        logger.info(f'测试视频 {i+1}/{len(video_files)}: {filename}')
        
        # 提取标签
        expected_label = extract_label_from_filename(filename)
        logger.info(f'预期标签: {expected_label}')
        
        try:
            # 测试视频
            detected_label = test_video(video_path, config.copy())
            logger.info(f'检测结果: {detected_label}')
            
            # 判断是否正确
            is_correct = expected_label == detected_label
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # 保存结果
            test_results.append({
                'video': filename,
                'expected_label': expected_label,
                'detected_label': detected_label,
                'is_correct': is_correct
            })
            
            logger.info(f'测试完成，结果: {"正确" if is_correct else "错误"}')
        except Exception as e:
            logger.error(f'测试视频 {filename} 失败: {e}')
            test_results.append({
                'video': filename,
                'expected_label': expected_label,
                'detected_label': '测试失败',
                'is_correct': False
            })
        
        logger.info('-' * 50)
    
    # 计算准确率
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        logger.info(f'测试完成，共测试 {total_count} 个视频')
        logger.info(f'正确 {correct_count} 个，错误 {total_count - correct_count} 个')
        logger.info(f'准确率: {accuracy:.2f}%')
    else:
        accuracy = 0
        logger.info('未测试任何视频')
    
    # 保存测试结果
    output_file = 'output/test_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_count': total_count,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'results': test_results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f'测试结果已保存至: {output_file}')
    
    return accuracy


if __name__ == '__main__':
    main()
