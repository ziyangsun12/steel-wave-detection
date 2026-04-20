#!/usr/bin/env python3
"""
带钢浪形检测系统 - 合并版
同时运行视频检测和Web服务器
"""

import os
import sys
import argparse
import threading
import time
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import Pipeline
from web_server import app

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/run_combined.log')
        ]
    )
    return logging.getLogger(__name__)

def run_pipeline(video_path, label):
    """运行检测Pipeline"""
    from utils import Utils
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = Utils.load_config(config_path)
    
    # 初始化Pipeline
    pipeline = Pipeline(config)
    
    # 启动处理线程
    pipeline.start()
    
    # 处理视频
    pipeline._read_video(video_path)
    
    # 等待处理完成
    while pipeline.running:
        time.sleep(1)

def run_web_server():
    """运行Web服务器"""
    app.run(host='0.0.0.0', port=5000, debug=False)

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='带钢浪形检测系统 - 合并版')
    parser.add_argument('--video', type=str, default=None, help='视频文件路径')
    parser.add_argument('--data-folder', type=str, default='data', help='数据文件夹路径')
    args = parser.parse_args()
    
    # 自动选择视频文件
    video_path = args.video
    label = 'Unknown'
    
    if not video_path:
        data_folder = args.data_folder
        if os.path.exists(data_folder):
            # 递归搜索视频文件
            video_files = []
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_files.append(os.path.join(root, file))
            
            if video_files:
                video_path = video_files[0]
                # 从文件名提取标签
                label = os.path.splitext(os.path.basename(video_path))[0]
                logger.info(f'自动选择视频文件: {video_path}')
                logger.info(f'视频标签: {label}')
            else:
                logger.error('未找到视频文件')
                return
        else:
            logger.error(f'数据文件夹不存在: {data_folder}')
            return
    else:
        # 从文件名提取标签
        label = os.path.splitext(os.path.basename(video_path))[0]
    
    # 启动检测Pipeline线程
    pipeline_thread = threading.Thread(target=run_pipeline, args=(video_path, label))
    pipeline_thread.daemon = True
    pipeline_thread.start()
    
    # 等待Pipeline启动
    time.sleep(2)
    
    # 启动Web服务器
    logger.info('启动带钢浪形检测Web服务器...')
    logger.info('访问地址: http://localhost:5000')
    logger.info('按 Ctrl+C 停止服务器')
    
    try:
        run_web_server()
    except KeyboardInterrupt:
        logger.info('停止带钢浪形检测系统')

if __name__ == '__main__':
    main()
