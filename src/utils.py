"""通用工具函数模块"""

import os
import yaml
import time
import json
import logging
from typing import Dict, Optional, Any


class Utils:
    """通用工具类"""
    
    def __init__(self, config: Dict):
        """初始化工具类
        
        Args:
            config: 配置参数
        """
        self.config = config
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置参数
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logging.error(f'加载配置文件失败: {e}')
            return {}
    
    @staticmethod
    def save_config(config: Dict, config_path: str):
        """保存配置文件
        
        Args:
            config: 配置参数
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logging.error(f'保存配置文件失败: {e}')
    
    @staticmethod
    def get_timestamp() -> str:
        """获取时间戳
        
        Returns:
            时间戳字符串
        """
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    @staticmethod
    def get_timestamp_ms() -> str:
        """获取毫秒级时间戳
        
        Returns:
            毫秒级时间戳字符串
        """
        return time.strftime('%Y-%m-%d %H:%M:%S.%f', time.localtime())[:-3]
    
    @staticmethod
    def mkdir_p(path: str):
        """创建目录（递归）
        
        Args:
            path: 目录路径
        """
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            logging.error(f'创建目录失败: {e}')
    
    @staticmethod
    def safe_json_load(file_path: str) -> Optional[Dict]:
        """安全加载JSON文件
        
        Args:
            file_path: JSON文件路径
        
        Returns:
            JSON数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logging.error(f'加载JSON文件失败: {e}')
            return None
    
    @staticmethod
    def safe_json_dump(data: Any, file_path: str):
        """安全保存JSON文件
        
        Args:
            data: 要保存的数据
            file_path: JSON文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f'保存JSON文件失败: {e}')
    
    @staticmethod
    def calculate_fps(frame_count: int, elapsed_time: float) -> float:
        """计算帧率
        
        Args:
            frame_count: 帧数
            elapsed_time:  elapsed时间
        
        Returns:
            帧率
        """
        if elapsed_time == 0:
            return 0.0
        return frame_count / elapsed_time
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """格式化时间
        
        Args:
            seconds: 秒数
        
        Returns:
            格式化的时间字符串
        """
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f'{int(hours)}h {int(minutes)}m {seconds:.2f}s'
        elif minutes > 0:
            return f'{int(minutes)}m {seconds:.2f}s'
        else:
            return f'{seconds:.2f}s'
    
    @staticmethod
    def validate_config(config: Dict, required_keys: list) -> bool:
        """验证配置
        
        Args:
            config: 配置参数
            required_keys: 必需的键
        
        Returns:
            配置是否有效
        """
        for key in required_keys:
            if key not in config:
                logging.error(f'配置缺少必需键: {key}')
                return False
        return True
    
    @staticmethod
    def get_file_list(directory: str, extensions: list = None) -> list:
        """获取目录中的文件列表
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名列表
        
        Returns:
            文件路径列表
        """
        file_list = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if extensions:
                        if any(file.endswith(ext) for ext in extensions):
                            file_list.append(os.path.join(root, file))
                    else:
                        file_list.append(os.path.join(root, file))
        except Exception as e:
            logging.error(f'获取文件列表失败: {e}')
        return file_list
    
    @staticmethod
    def get_relative_path(path: str, base: str) -> str:
        """获取相对路径
        
        Args:
            path: 绝对路径
            base: 基准路径
        
        Returns:
            相对路径
        """
        try:
            return os.path.relpath(path, base)
        except Exception as e:
            logging.error(f'获取相对路径失败: {e}')
            return path
    
    @staticmethod
    def format_size(size: int) -> str:
        """格式化文件大小
        
        Args:
            size: 文件大小（字节）
        
        Returns:
            格式化的大小字符串
        """
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return f'{size:.2f} {units[unit_index]}'
    
    @staticmethod
    def exception_handler(func):
        """异常处理装饰器
        
        Args:
            func: 要装饰的函数
        
        Returns:
            装饰后的函数
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f'函数 {func.__name__} 执行失败: {e}')
                return None
        return wrapper
    
    @staticmethod
    def timing_decorator(func):
        """计时装饰器
        
        Args:
            func: 要装饰的函数
        
        Returns:
            装饰后的函数
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            logging.debug(f'函数 {func.__name__} 执行时间: {elapsed:.4f}s')
            return result
        return wrapper
    
    @staticmethod
    def singleton(cls):
        """单例装饰器
        
        Args:
            cls: 要装饰的类
        
        Returns:
            装饰后的类
        """
        instances = {}
        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        return get_instance
    
    def get_camera_params(self) -> Dict:
        """获取相机参数
        
        Returns:
            相机参数
        """
        return self.config.get('camera', {})
    
    def get_calibration_params(self) -> Dict:
        """获取标定参数
        
        Returns:
            标定参数
        """
        return self.config.get('calibration', {})
    
    def get_preprocessing_params(self) -> Dict:
        """获取预处理参数
        
        Returns:
            预处理参数
        """
        return self.config.get('preprocessing', {})
    
    def get_reconstruction_params(self) -> Dict:
        """获取重建参数
        
        Returns:
            重建参数
        """
        return self.config.get('reconstruction', {})
    
    def get_detection_params(self) -> Dict:
        """获取检测参数
        
        Returns:
            检测参数
        """
        return self.config.get('detection', {})
    
    def get_visualization_params(self) -> Dict:
        """获取可视化参数
        
        Returns:
            可视化参数
        """
        return self.config.get('visualization', {})
    
    def get_output_params(self) -> Dict:
        """获取输出参数
        
        Returns:
            输出参数
        """
        return self.config.get('output', {})
