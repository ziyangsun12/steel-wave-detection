"""带钢浪形检测实时处理Pipeline"""

import cv2
import numpy as np
import threading
import queue
import time
import json
import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

from src.preprocessing import Preprocessor
from src.reconstruction_3d import Reconstructor
from src.wave_detector import WaveDetector
from src.visualization import Visualizer
from src.utils import Utils


class Pipeline:
    """带钢浪形检测实时处理Pipeline"""
    
    def __init__(self, config: Dict):
        """初始化Pipeline
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.realtime_config = config.get('realtime', {})
        self.output_config = config.get('output', {})
        
        # 初始化各模块
        self.preprocessor = Preprocessor(config)
        self.reconstructor = Reconstructor(config)
        self.detector = WaveDetector(config)
        self.visualizer = Visualizer(config)
        self.utils = Utils(config)
        
        # 初始化队列
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        
        # 初始化线程
        self.threads = []
        self.running = False
        
        # 性能监控
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        
        # 数据输出
        self.csv_writer = None
        self.csv_file = None
        self.json_file = None
        
        # 初始化日志
        self._init_logging()
        
    def _init_logging(self):
        """初始化日志"""
        logging_config = self.config.get('logging', {})
        level = getattr(logging, logging_config.get('level', 'INFO'), logging.INFO)
        log_file = logging_config.get('file', 'logs/detection.log')
        log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 确保日志路径是绝对路径
        if not os.path.isabs(log_file):
            log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log_file)
        
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_output(self):
        """初始化数据输出"""
        # 项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 初始化CSV输出
        if self.output_config.get('csv', {}).get('enable', True):
            csv_file = self.output_config.get('csv', {}).get('file', 'output/detection_results.csv')
            try:
                # 确保CSV路径是绝对路径
                if not os.path.isabs(csv_file):
                    csv_file = os.path.join(project_root, csv_file)
                
                # 确保输出目录存在
                csv_dir = os.path.dirname(csv_file)
                if csv_dir and not os.path.exists(csv_dir):
                    os.makedirs(csv_dir, exist_ok=True)
                
                self.csv_file = open(csv_file, 'w', newline='', encoding='utf-8')
                fieldnames = ['timestamp', 'wave_type', 'wave_level', 'wave_height', 'wave_width', 'wave_position_x', 'wave_position_y']
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
                self.csv_writer.writeheader()
                self.logger.info(f'CSV输出已初始化: {csv_file}')
            except Exception as e:
                self.logger.error(f'初始化CSV输出失败: {e}')
        
        # 初始化JSON输出
        if self.output_config.get('json', {}).get('enable', True):
            json_file = self.output_config.get('json', {}).get('file', 'output/detection_results.json')
            try:
                # 确保JSON路径是绝对路径
                if not os.path.isabs(json_file):
                    json_file = os.path.join(project_root, json_file)
                
                # 确保输出目录存在
                json_dir = os.path.dirname(json_file)
                if json_dir and not os.path.exists(json_dir):
                    os.makedirs(json_dir, exist_ok=True)
                
                self.json_file = open(json_file, 'w', encoding='utf-8')
                self.json_file.write('[\n')
                self.logger.info(f'JSON输出已初始化: {json_file}')
            except Exception as e:
                self.logger.error(f'初始化JSON输出失败: {e}')
    
    def _close_output(self):
        """关闭数据输出"""
        if self.csv_file:
            try:
                self.csv_file.close()
                self.logger.info('CSV输出已关闭')
            except Exception as e:
                self.logger.error(f'关闭CSV输出失败: {e}')
        
        if self.json_file:
            try:
                self.json_file.write(']\n')
                self.json_file.close()
                self.logger.info('JSON输出已关闭')
            except Exception as e:
                self.logger.error(f'关闭JSON输出失败: {e}')
    
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像
        
        Args:
            frame: 输入帧图像
        
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 预处理
        binary, preprocess_info = self.preprocessor.preprocess(frame)
        
        # 分割带钢
        mask, segment_info = self.preprocessor.segment_steel(binary)
        
        # 三维重建
        contour_data, reconstruct_info = self.reconstructor.reconstruct(binary, mask)
        
        # 浪形检测
        detection_result, detect_info = self.detector.detect(contour_data)
        
        # 计算性能
        process_time = time.time() - start_time
        
        # 更新FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        # 构建结果
        result = {
            'frame': frame,
            'binary': binary,
            'mask': mask,
            'contour_data': contour_data,
            'detection_result': detection_result,
            'info': {
                'preprocess': preprocess_info,
                'segment': segment_info,
                'reconstruct': reconstruct_info,
                'detect': detect_info,
                'process_time': process_time,
                'fps': self.fps
            }
        }
        
        return result
    
    def _process_thread(self):
        """处理线程"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
                
                result = self._process_frame(frame)
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f'处理线程错误: {e}')
                continue
    
    def _visualize_thread(self):
        """可视化线程"""
        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)
                if result is None:
                    break
                
                # 可视化
                vis_frame = self.visualizer.visualize(
                    result['frame'],
                    result['contour_data'],
                    result['detection_result'],
                    result['info']
                )
                
                # 显示
                if not self.visualizer.show(vis_frame):
                    self.running = False
                
                # 保存结果
                self._save_result(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f'可视化线程错误: {e}')
                continue
    
    def _save_result(self, result: Dict):
        """保存检测结果
        
        Args:
            result: 处理结果
        """
        detection_result = result['detection_result']
        timestamp = time.time()
        
        # 保存到CSV
        if self.csv_writer:
            try:
                row = {
                    'timestamp': timestamp,
                    'wave_type': detection_result.get('wave_type', '未知'),
                    'wave_level': detection_result.get('wave_level', '低'),
                    'wave_height': detection_result.get('wave_height', 0),
                    'wave_width': detection_result.get('wave_width', 0),
                    'wave_position_x': detection_result.get('wave_position', {}).get('x', 0),
                    'wave_position_y': detection_result.get('wave_position', {}).get('y', 0)
                }
                self.csv_writer.writerow(row)
                self.csv_file.flush()
            except Exception as e:
                self.logger.error(f'保存到CSV失败: {e}')
        
        # 保存到JSON
        if self.json_file:
            try:
                json_data = {
                    'timestamp': timestamp,
                    'detection_result': detection_result,
                    'info': result['info']
                }
                json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                self.json_file.write(json_str + ',\n')
                self.json_file.flush()
            except Exception as e:
                self.logger.error(f'保存到JSON失败: {e}')
    
    def _read_video(self, video_path: str):
        """读取视频
        
        Args:
            video_path: 视频路径
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f'无法打开视频: {video_path}')
            return
        
        self.logger.info(f'开始处理视频: {video_path}')
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 限制队列大小
            if self.frame_queue.qsize() < self.frame_queue.maxsize:
                self.frame_queue.put(frame)
            else:
                # 丢弃旧帧
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass
        
        cap.release()
        self.logger.info('视频处理完成')
    
    def _read_camera(self, camera_id: int):
        """读取相机
        
        Args:
            camera_id: 相机ID
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error(f'无法打开相机: {camera_id}')
            return
        
        # 设置相机参数
        width = self.config.get('camera', {}).get('resolution', {}).get('width', 1280)
        height = self.config.get('camera', {}).get('resolution', {}).get('height', 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.logger.info(f'开始处理相机: {camera_id}')
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # 相机断流处理
                self.logger.warning('相机断流，尝试重连...')
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(camera_id)
                continue
            
            # 限制队列大小
            if self.frame_queue.qsize() < self.frame_queue.maxsize:
                self.frame_queue.put(frame)
            else:
                # 丢弃旧帧
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass
        
        cap.release()
        self.logger.info('相机处理完成')
    
    def _read_folder(self, folder_path: str):
        """读取文件夹中的图像
        
        Args:
            folder_path: 文件夹路径
        """
        import os
        
        # 获取文件夹中的图像文件
        image_files = []
        for file in os.listdir(folder_path):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(folder_path, file))
        
        image_files.sort()
        self.logger.info(f'开始处理文件夹: {folder_path}, 共{len(image_files)}张图像')
        
        for image_file in image_files:
            if not self.running:
                break
            
            frame = cv2.imread(image_file)
            if frame is None:
                self.logger.warning(f'无法读取图像: {image_file}')
                continue
            
            # 限制队列大小
            if self.frame_queue.qsize() < self.frame_queue.maxsize:
                self.frame_queue.put(frame)
            else:
                # 丢弃旧帧
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass
            
            # 控制处理速度
            time.sleep(0.1)
        
        self.logger.info('文件夹处理完成')
    
    def start(self):
        """启动Pipeline"""
        self.logger.info('启动带钢浪形检测Pipeline')
        self.running = True
        
        # 初始化输出
        self._init_output()
        
        # 启动处理线程
        num_threads = self.realtime_config.get('multithreading', {}).get('num_threads', 4)
        for i in range(num_threads):
            t = threading.Thread(target=self._process_thread)
            t.daemon = True
            t.start()
            self.threads.append(t)
        
        # 启动可视化线程
        t = threading.Thread(target=self._visualize_thread)
        t.daemon = True
        t.start()
        self.threads.append(t)
        
        # 启动视频/相机读取
        camera_config = self.config.get('camera', {})
        camera_type = camera_config.get('type', 'local')
        
        if camera_type == 'local':
            video_path = camera_config.get('video_path', 'data/sample_video.mp4')
            self._read_video(video_path)
        elif camera_type == 'industrial':
            camera_id = camera_config.get('camera_id', 0)
            self._read_camera(camera_id)
        elif camera_type == 'folder':
            folder_path = camera_config.get('folder_path', 'data/images')
            self._read_folder(folder_path)
        
    def stop(self):
        """停止Pipeline"""
        self.logger.info('停止带钢浪形检测Pipeline')
        self.running = False
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # 等待线程结束
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)
        
        # 关闭输出
        self._close_output()
        
        # 关闭可视化
        self.visualizer.close()
        
        self.logger.info('Pipeline已停止')
    
    def get_status(self):
        """获取Pipeline状态
        
        Returns:
            状态信息字典
        """
        return {
            'running': self.running,
            'fps': round(self.fps, 2),
            'frame_queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'thread_count': len(self.threads),
            'camera_type': self.config.get('camera', {}).get('type', 'local'),
            'video_path': self.config.get('camera', {}).get('video_path', '')
        }
