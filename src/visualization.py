"""带钢浪形可视化模块"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class Visualizer:
    """带钢浪形可视化类"""
    
    def __init__(self, config: Dict):
        """初始化可视化参数
        
        Args:
            config: 可视化配置参数
        """
        self.config = config
        self.vis_config = config.get('visualization', {})
        self.display_config = self.vis_config.get('display', {})
        self.drawing_config = self.vis_config.get('drawing', {})
        
        # 颜色配置
        self.colors = self.drawing_config.get('colors', {
            'contour': [0, 255, 0],
            'wave': [0, 0, 255],
            'text': [255, 255, 255]
        })
        
        # 线宽和字体大小
        self.line_width = self.drawing_config.get('line_width', 2)
        self.font_size = self.drawing_config.get('font_size', 1.0)
        
        # 窗口配置
        self.window_name = self.display_config.get('window_name', 'Steel Wave Detection')
        self.window_size = self.display_config.get('window_size', {'width': 1280, 'height': 720})
        
    def visualize(self, frame: np.ndarray, contour_data: Dict, detection_result: Dict, info: Dict) -> np.ndarray:
        """可视化检测结果
        
        Args:
            frame: 原始帧图像
            contour_data: 轮廓数据
            detection_result: 检测结果
            info: 处理信息
        
        Returns:
            可视化后的图像
        """
        # 创建副本（使用较小的尺寸以减少内存占用）
        target_width = self.window_size.get('width', 1280)
        target_height = self.window_size.get('height', 720)
        
        # 如果图像太大，先缩小
        if frame.shape[1] > target_width or frame.shape[0] > target_height:
            scale = min(target_width / frame.shape[1], target_height / frame.shape[0])
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            vis_frame = cv2.resize(frame, (new_width, new_height))
        else:
            vis_frame = frame.copy()
        
        # 绘制轮廓
        if self.drawing_config.get('enable', True):
            # 绘制YOLOv8分割结果（如果有）
            if 'segment' in info and 'steel_bbox' in info['segment']:
                vis_frame = self._draw_yolo_segmentation(vis_frame, info['segment'])
            
            vis_frame = self._draw_contours(vis_frame, contour_data)
            vis_frame = self._draw_lines(vis_frame, contour_data)
            vis_frame = self._draw_wave_info(vis_frame, detection_result, info)
        
        return vis_frame
    
    def _draw_contours(self, frame: np.ndarray, contour_data: Dict) -> np.ndarray:
        """绘制轮廓
        
        Args:
            frame: 输入帧图像
            contour_data: 轮廓数据
        
        Returns:
            绘制后的图像
        """
        # 绘制带钢边界
        if 'left_edge' in contour_data and 'right_edge' in contour_data:
            left_edge = contour_data['left_edge']
            right_edge = contour_data['right_edge']
            
            if len(left_edge) > 0:
                cv2.polylines(frame, [left_edge], False, self.colors['contour'], self.line_width)
            if len(right_edge) > 0:
                cv2.polylines(frame, [right_edge], False, self.colors['contour'], self.line_width)
        
        return frame
    
    def _draw_lines(self, frame: np.ndarray, contour_data: Dict) -> np.ndarray:
        """绘制中线和浪形
        
        Args:
            frame: 输入帧图像
            contour_data: 轮廓数据
        
        Returns:
            绘制后的图像
        """
        # 绘制中线
        if 'center_line' in contour_data:
            center_line = contour_data['center_line']
            if len(center_line) > 0:
                cv2.polylines(frame, [center_line], False, (255, 0, 0), self.line_width)
        
        # 绘制浪形
        if 'height_diff' in contour_data:
            height_diff = contour_data['height_diff']
            center_line = contour_data.get('center_line', [])
            
            if len(height_diff) > 0 and len(center_line) > 0:
                # 找到波峰和波谷
                peaks = self._find_peaks(height_diff)
                valleys = self._find_valleys(height_diff)
                
                # 绘制波峰
                for peak in peaks:
                    if peak < len(center_line):
                        cv2.circle(frame, tuple(center_line[peak]), 5, self.colors['wave'], -1)
                
                # 绘制波谷
                for valley in valleys:
                    if valley < len(center_line):
                        cv2.circle(frame, tuple(center_line[valley]), 3, (0, 255, 255), -1)
        
        return frame
    
    def _draw_yolo_segmentation(self, frame: np.ndarray, segment_info: Dict) -> np.ndarray:
        """绘制YOLOv8分割结果
        
        Args:
            frame: 输入帧图像
            segment_info: 分割信息
        
        Returns:
            绘制后的图像
        """
        # 绘制卷钢边界框
        if 'steel_bbox' in segment_info:
            x, y, w, h = segment_info['steel_bbox']
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 绘制标签
            cv2.putText(frame, 'Steel Coil', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 绘制分割掩码（如果有）
        if 'mask' in segment_info:
            mask = segment_info['mask']
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                # 创建掩码的彩色版本
                mask_color = np.zeros_like(frame)
                mask_color[:, :, 0] = mask  # 蓝色通道
                # 叠加掩码到原始图像
                frame = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
        
        return frame
    
    def _draw_wave_info(self, frame: np.ndarray, detection_result: Dict, info: Dict) -> np.ndarray:
        """绘制浪形信息
        
        Args:
            frame: 输入帧图像
            detection_result: 检测结果
            info: 处理信息
        
        Returns:
            绘制后的图像
        """
        # 绘制浪形类型
        wave_type = detection_result.get('wave_type', 'Unknown')
        wave_level = detection_result.get('wave_level', 'Low')
        wave_height = detection_result.get('wave_height', 0)
        wave_width = detection_result.get('wave_width', 0)
        
        # 绘制文本
        text_y = 30
        text_step = 30
        
        cv2.putText(frame, f'Type: {wave_type}', (20, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.colors['text'], 2)
        text_y += text_step
        
        cv2.putText(frame, f'Level: {wave_level}', (20, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.colors['text'], 2)
        text_y += text_step
        
        cv2.putText(frame, f'Height: {wave_height} mm', (20, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.colors['text'], 2)
        text_y += text_step
        
        cv2.putText(frame, f'Width: {wave_width} mm', (20, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.colors['text'], 2)
        text_y += text_step
        
        # 绘制浪形位置
        wave_position = detection_result.get('wave_position', {})
        if wave_position:
            cv2.putText(frame, f'Pos: ({wave_position.get("x", 0):.1f}, {wave_position.get("y", 0):.1f})', 
                        (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.colors['text'], 2)
            text_y += text_step
        
        # 绘制处理信息
        if 'fps' in info:
            cv2.putText(frame, f'FPS: {info["fps"]:.1f}', (frame.shape[1] - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.colors['text'], 2)
        
        if 'overexposed' in info and info['overexposed']:
            cv2.putText(frame, 'OVEREXPOSED', (frame.shape[1] - 200, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 0, 255), 2)
        
        if 'black_screen' in info and info['black_screen']:
            cv2.putText(frame, 'BLACK SCREEN', (frame.shape[1] - 200, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 0, 255), 2)
        
        # 显示异常信息
        if 'exception' in info:
            exception_info = info['exception']
            if exception_info.get('has_exception', False):
                exception_message = exception_info.get('message', '异常')
                # 限制消息长度，避免显示过长
                if len(exception_message) > 30:
                    exception_message = exception_message[:30] + '...'
                cv2.putText(frame, f'异常: {exception_message}', (20, frame.shape[0] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 0, 255), 2)
        
        return frame
    
    def show(self, frame: np.ndarray) -> bool:
        """显示图像
        
        Args:
            frame: 要显示的图像
        
        Returns:
            是否继续显示
        """
        if not self.display_config.get('enable', True):
            return True
        
        try:
            # 尝试显示图像，在没有图形界面的环境中会失败
            # 调整窗口大小
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.window_size['width'], self.window_size['height'])
            
            # 显示图像
            cv2.imshow(self.window_name, frame)
            
            # 检查按键（使用较短的等待时间，避免阻塞）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('s'):
                # 保存当前帧
                try:
                    cv2.imwrite('output/capture.jpg', frame)
                    print('图像已保存')
                except Exception as e:
                    print(f'保存图像失败: {e}')
        except Exception as e:
            # 在没有图形界面的环境中，跳过显示
            pass
        
        return True
    
    def plot_height_profile(self, contour_data: Dict, save_path: Optional[str] = None):
        """绘制高度轮廓图
        
        Args:
            contour_data: 轮廓数据
            save_path: 保存路径
        """
        if 'height_diff' not in contour_data:
            return
        
        height_diff = contour_data['height_diff']
        
        plt.figure(figsize=(12, 6))
        plt.plot(height_diff)
        plt.title('带钢高度轮廓')
        plt.xlabel('位置')
        plt.ylabel('高度差 (mm)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def plot_3d(self, contour_data: Dict, save_path: Optional[str] = None):
        """绘制三维点云
        
        Args:
            contour_data: 轮廓数据
            save_path: 保存路径
        """
        if 'height_field' not in contour_data:
            return
        
        height_field = contour_data['height_field']
        if len(height_field) == 0:
            return
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = height_field[:, 1]
        y = height_field[:, 0]
        z = height_field[:, 2]
        
        ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.set_title('带钢三维轮廓')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def save_result(self, frame: np.ndarray, detection_result: Dict, save_path: str):
        """保存检测结果
        
        Args:
            frame: 可视化后的图像
            detection_result: 检测结果
            save_path: 保存路径
        """
        # 保存图像
        cv2.imwrite(save_path, frame)
        
        # 保存结果到文本文件
        result_path = save_path.replace('.jpg', '.txt').replace('.png', '.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write('带钢浪形检测结果\n')
            f.write('-' * 50 + '\n')
            f.write(f'浪形类型: {detection_result.get("wave_type", "未知")}\n')
            f.write(f'浪形等级: {detection_result.get("wave_level", "低")}\n')
            f.write(f'浪高: {detection_result.get("wave_height", 0)} mm\n')
            f.write(f'浪宽: {detection_result.get("wave_width", 0)} mm\n')
            if 'wave_position' in detection_result:
                pos = detection_result['wave_position']
                f.write(f'浪形位置: ({pos.get("x", 0):.2f}, {pos.get("y", 0):.2f}) mm\n')
    
    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """查找波峰
        
        Args:
            data: 数据序列
        
        Returns:
            波峰索引列表
        """
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return peaks
    
    def _find_valleys(self, data: np.ndarray) -> List[int]:
        """查找波谷
        
        Args:
            data: 数据序列
        
        Returns:
            波谷索引列表
        """
        valleys = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                valleys.append(i)
        return valleys
    
    def close(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()
