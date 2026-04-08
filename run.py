"""带钢浪形检测系统启动脚本"""

import os
import sys
import logging
import argparse
from src.utils import Utils
from src.pipeline import Pipeline


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='带钢浪形检测系统')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--video', type=str, default=None, help='视频文件路径')
    parser.add_argument('--camera', type=int, default=None, help='相机ID')
    parser.add_argument('--folder', type=str, default=None, help='图像文件夹路径')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    return parser.parse_args()


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 解析命令行参数
    args = parse_args()
    
    # 开启调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 加载配置文件
    logger.info(f'加载配置文件: {args.config}')
    config = Utils.load_config(args.config)
    if not config:
        logger.error('加载配置文件失败，退出程序')
        sys.exit(1)
    
    # 覆盖配置（如果命令行参数提供）
    if args.video:
        config['camera']['type'] = 'local'
        config['camera']['video_path'] = args.video
    elif args.camera is not None:
        config['camera']['type'] = 'industrial'
        config['camera']['camera_id'] = args.camera
    elif args.folder:
        config['camera']['type'] = 'folder'
        config['camera']['folder_path'] = args.folder
    
    # 确保输出目录存在
    output_dirs = ['output', 'logs']
    for dir_name in output_dirs:
        Utils.mkdir_p(dir_name)
    
    # 初始化Pipeline
    logger.info('初始化带钢浪形检测Pipeline')
    pipeline = Pipeline(config)
    
    try:
        # 启动Pipeline
        logger.info('启动带钢浪形检测系统')
        pipeline.start()
    except KeyboardInterrupt:
        logger.info('用户中断，停止系统')
    except Exception as e:
        logger.error(f'系统运行失败: {e}')
    finally:
        # 停止Pipeline
        logger.info('停止带钢浪形检测系统')
        pipeline.stop()


if __name__ == '__main__':
    main()
