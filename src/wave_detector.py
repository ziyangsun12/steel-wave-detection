"""带钢浪形检测模块"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

# 尝试导入YOLOv8模型
try:
    from ultralytics import YOLO
    has_yolo = True
except ImportError:
    has_yolo = False


class WaveDetector:
    """带钢浪形检测类"""
    
    def __init__(self, config: Dict = None):
        """初始化检测参数
        
        Args:
            config: 检测配置参数
        """
        self.config = config or {}
        self.detect_config = self.config.get('detection', {})
        
        # 浪形等级阈值
        self.wave_levels = self.detect_config.get('wave_levels', {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0
        })
        
        # 加载YOLOv8模型（如果使用）
        self.yolo_model = None
        # 加载预训练的YOLOv8模型
        if has_yolo:
            try:
                import os
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8_steel_coil.pt')
                if os.path.exists(model_path):
                    self.yolo_model = YOLO(model_path)
                    print(f"成功加载YOLO模型: {model_path}")
                else:
                    print(f"YOLO模型文件不存在: {model_path}")
            except Exception as e:
                print(f"加载YOLO模型失败: {e}")
        
        # 时序优化参数
        self.history_size = 10  # 历史帧数量
        self.history = []  # 历史检测结果
        self.optical_flow = None  # 光流对象
        self.last_frame = None  # 上一帧图像
    
    def detect(self, contour_data: Dict, frame: Optional[np.ndarray] = None) -> Tuple[Dict, Dict]:
        """检测浪形
        
        Args:
            contour_data: 轮廓数据
            frame: 当前帧图像（用于光流计算）
        
        Returns:
            浪形检测结果和检测信息
        """
        info = {}
        
        # 检查输入
        if not contour_data:
            info['error'] = '无效输入'
            return {}, info
        
        # 计算浪形参数
        wave_params = self._calculate_wave_parameters(contour_data)
        info.update(wave_params)
        
        # 分类浪形类型
        wave_type = self._classify_wave_type(contour_data, wave_params)
        info['wave_type'] = wave_type
        
        # 分级浪形
        wave_level = self._classify_wave_level(wave_params.get('wave_height', 0))
        info['wave_level'] = wave_level
        
        # 检测复合浪形
        composite_wave = self._detect_composite_wave(contour_data, wave_params)
        if composite_wave:
            info['composite_wave'] = composite_wave
        
        # 构建检测结果
        result = {
            'wave_type': wave_type,
            'wave_level': wave_level,
            'wave_height': wave_params.get('wave_height', 0),
            'wave_width': wave_params.get('wave_width', 0),
            'wave_position': wave_params.get('wave_position', {}),
            'composite_wave': composite_wave
        }
        
        # 时序优化
        if frame is not None:
            result, info = self._temporal_optimization(result, frame, info)
        
        # 保存到历史记录
        self._update_history(result)
        
        return result, info
    
    def _calculate_wave_parameters(self, contour_data: Dict) -> Dict:
        """计算浪形参数
        
        Args:
            contour_data: 轮廓数据
        
        Returns:
            浪形参数
        """
        params = {}
        
        # 从轮廓数据中获取高度差
        height_diff = contour_data.get('height_diff', [])
        if len(height_diff) == 0:
            return params
        
        # 计算浪高
        max_height = np.max(height_diff)
        min_height = np.min(height_diff)
        wave_height = max_height - min_height
        params['wave_height'] = round(wave_height, 2)
        
        # 计算浪宽
        center_line = contour_data.get('center_line', [])
        if len(center_line) > 0:
            # 找到波峰和波谷位置
            peaks = self._find_peaks(height_diff)
            valleys = self._find_valleys(height_diff)
            
            if len(peaks) > 1:
                # 计算平均浪宽
                peak_positions = np.array(peaks) * self.config.get('calibration', {}).get('pixel_to_mm', {}).get('y', 0.11)
                wave_widths = np.diff(peak_positions)
                avg_wave_width = np.mean(wave_widths)
                params['wave_width'] = round(avg_wave_width, 2)
            
            # 计算浪形位置
            if peaks:
                max_peak_idx = peaks[np.argmax(height_diff[peaks])]
                max_peak_pos = center_line[max_peak_idx]
                params['wave_position'] = {
                    'x': round(max_peak_pos[0] * self.config.get('calibration', {}).get('pixel_to_mm', {}).get('x', 0.1), 2),
                    'y': round(max_peak_pos[1] * self.config.get('calibration', {}).get('pixel_to_mm', {}).get('y', 0.11), 2)
                }
        
        return params
    
    def _classify_wave_type(self, contour_data: Dict, wave_params: Dict) -> str:
        """分类浪形类型
        
        Args:
            contour_data: 轮廓数据
            wave_params: 浪形参数
        
        Returns:
            浪形类型
        """
        # 检查是否有足够的数据
        height_diff = contour_data.get('height_diff', [])
        if len(height_diff) == 0:
            return '未知'
        
        # 计算浪高
        wave_height = wave_params.get('wave_height', 0)
        if wave_height < self.detect_config.get('parameters', {}).get('min_wave_height', 0.5):
            return '平直'
        
        # 分析高度分布
        center_line = contour_data.get('center_line', [])
        left_edge = contour_data.get('left_edge', [])
        right_edge = contour_data.get('right_edge', [])
        
        if len(center_line) == 0 or len(left_edge) == 0 or len(right_edge) == 0:
            return '未知'
        
        # 计算左右边缘的高度差异
        left_width = np.linalg.norm(right_edge - left_edge, axis=1)
        right_width = np.linalg.norm(right_edge - left_edge, axis=1)
        
        # 计算宽度变化率
        width_change = np.diff(left_width)
        if len(width_change) == 0:
            return '未知'
        
        # 分析宽度变化模式
        max_change = np.max(np.abs(width_change))
        avg_change = np.mean(np.abs(width_change))
        
        # 基于变化模式分类浪形
        if max_change < 0.5:
            return '平直'
        elif self._is_edge_wave(left_width, right_width):
            # 进一步判断是DS还是WS单边浪
            if self._is_ds_edge_wave(left_width):
                return 'DS单边浪'
            else:
                return 'WS单边浪'
        elif self._is_center_wave(left_width):
            return '中浪'
        elif self._is_double_edge_wave(left_width, right_width):
            return '双边浪'
        else:
            return '未知'
    
    def _classify_wave_level(self, wave_height: float) -> str:
        """分级浪形
        
        Args:
            wave_height: 浪高
        
        Returns:
            浪形等级
        """
        if wave_height < self.wave_levels.get('low', 1.0):
            return '低'
        elif wave_height < self.wave_levels.get('medium', 2.0):
            return '中'
        elif wave_height < self.wave_levels.get('high', 3.0):
            return '高'
        else:
            return '严重'
    
    def _detect_composite_wave(self, contour_data: Dict, wave_params: Dict) -> List[str]:
        """检测复合浪形
        
        Args:
            contour_data: 轮廓数据
            wave_params: 浪形参数
        
        Returns:
            复合浪形类型列表
        """
        composite = []
        
        # 这里可以实现复合浪形的检测逻辑
        # 例如同时检测到边浪和中浪
        
        return composite
    
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
    
    def _is_edge_wave(self, left_width: np.ndarray, right_width: np.ndarray) -> bool:
        """判断是否为边浪
        
        Args:
            left_width: 左边宽度序列
            right_width: 右边宽度序列
        
        Returns:
            是否为边浪
        """
        # 计算左右宽度的差异
        width_diff = np.abs(left_width - right_width)
        avg_diff = np.mean(width_diff)
        
        return avg_diff > 0.5
    
    def _is_ds_edge_wave(self, width: np.ndarray) -> bool:
        """判断是否为DS单边浪
        
        Args:
            width: 宽度序列
        
        Returns:
            是否为DS单边浪
        """
        # DS单边浪通常表现为宽度从左到右逐渐增加
        trend = np.polyfit(range(len(width)), width, 1)[0]
        return trend > 0.01
    
    def _is_center_wave(self, width: np.ndarray) -> bool:
        """判断是否为中浪
        
        Args:
            width: 宽度序列
        
        Returns:
            是否为中浪
        """
        # 中浪通常表现为宽度中间大两边小
        # 拟合二次曲线
        if len(width) < 3:
            return False
        
        coeffs = np.polyfit(range(len(width)), width, 2)
        # 二次项系数为负表示中间高两边低
        return coeffs[0] < -0.001
    
    def _is_double_edge_wave(self, left_width: np.ndarray, right_width: np.ndarray) -> bool:
        """判断是否为双边浪
        
        Args:
            left_width: 左边宽度序列
            right_width: 右边宽度序列
        
        Returns:
            是否为双边浪
        """
        # 双边浪通常表现为左右两边宽度都有较大变化
        left_change = np.max(np.abs(np.diff(left_width)))
        right_change = np.max(np.abs(np.diff(right_width)))
        
        return left_change > 0.5 and right_change > 0.5
    
    def validate_detection(self, result: Dict) -> bool:
        """验证检测结果
        
        Args:
            result: 检测结果
        
        Returns:
            检测结果是否有效
        """
        # 检查浪高是否在合理范围内
        wave_height = result.get('wave_height', 0)
        if wave_height < 0 or wave_height > 50:  # 50mm为合理最大值
            return False
        
        # 检查浪宽是否在合理范围内
        wave_width = result.get('wave_width', 0)
        if wave_width < 0 or wave_width > 1000:  # 1000mm为合理最大值
            return False
        
        return True
    
    def detect_from_frame(self, frame: np.ndarray) -> tuple:
        """从帧图像检测浪形
        
        Args:
            frame: 输入帧图像
        
        Returns:
            (浪形类型, 浪高, 水平偏差)
        """
        # 使用YOLOv8模型检测卷钢
        steel_coil_region = None
        if self.yolo_model:
            try:
                results = self.yolo_model(frame)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = box.conf[0]
                        if confidence > 0.5:
                            # 提取卷钢区域
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            steel_coil_region = frame[y1:y2, x1:x2]
                            break
            except Exception as e:
                print(f"YOLO检测失败: {e}")
        
        # 如果YOLO检测失败，使用颜色-based检测作为 fallback
        if steel_coil_region is None:
            # 颜色-based检测
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([150, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 | mask2
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > 100 and h > 50:
                    steel_coil_region = frame[y:y+h, x:x+w]
        
        # 如果找到了卷钢区域，进行边缘检测和浪形分析
        if steel_coil_region is not None:
            # 边缘检测
            edges = self._detect_edges(steel_coil_region)
            
            # 形状分析
            wave_type, wave_height, h_err = self._analyze_shape(edges, steel_coil_region.shape)
            
            return wave_type, wave_height, h_err
        else:
            # 未检测到卷钢
            return '未检测到卷钢', 0, 0
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """检测图像边缘
        
        Args:
            image: 输入图像
        
        Returns:
            边缘图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blur, 50, 150)
        
        # 膨胀操作，连接断裂的边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def _analyze_shape(self, edges: np.ndarray, image_shape: tuple) -> tuple:
        """分析边缘形状，检测浪形
        
        Args:
            edges: 边缘图像
            image_shape: 图像形状
        
        Returns:
            (浪形类型, 浪高, 水平偏差)
        """
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return '平直', 0, 0
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 计算轮廓的高度变化
        height_profile = []
        for i in range(x, x + w, 10):
            # 提取垂直线上的边缘点
            line = edges[:, i]
            edge_points = np.where(line > 0)[0]
            if len(edge_points) > 0:
                # 计算该列的高度（最大y - 最小y）
                height = np.max(edge_points) - np.min(edge_points)
                height_profile.append(height)
            else:
                height_profile.append(0)
        
        # 计算浪高
        if height_profile:
            max_height = np.max(height_profile)
            min_height = np.min(height_profile)
            # 使用配置文件中的像素到毫米转换参数
            pixel_to_mm = self.config.get('calibration', {}).get('pixel_to_mm', {}).get('y', 0.11)
            wave_height = (max_height - min_height) * pixel_to_mm  # 转换为毫米
            wave_height = round(wave_height, 2)
        else:
            wave_height = 0
        
        # 分析浪形类型
        if wave_height < 0.5:
            return '平直', wave_height, 0
        
        # 计算水平偏差
        # 假设卷钢应该是水平的，计算轮廓的倾斜度
        [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.arctan2(vy, vx) * 180 / np.pi
        h_err = round(angle, 2)
        
        # 分析高度分布，判断浪形类型
        if len(height_profile) > 3:
            # 计算高度分布的标准差
            std_height = np.std(height_profile)
            
            # 计算高度分布的趋势
            trend = np.polyfit(range(len(height_profile)), height_profile, 1)[0]
            
            # 计算高度分布的二次曲线系数
            coeffs = np.polyfit(range(len(height_profile)), height_profile, 2)
            
            # 判断浪形类型
            if std_height < 10:
                return '平直', wave_height, h_err
            elif abs(trend) > 0.5:
                # 单边浪
                if trend > 0:
                    return 'DS单边浪', wave_height, h_err
                else:
                    return 'WS单边浪', wave_height, h_err
            elif coeffs[0] < -0.01:
                # 中浪（中间高两边低）
                return '中浪', wave_height, h_err
            elif coeffs[0] > 0.01:
                # 双边浪（两边高中间低）
                return '双边浪', wave_height, h_err
            else:
                return '平直', wave_height, h_err
        else:
            return '平直', wave_height, h_err
    
    def get_wave_info(self, shape: str, height: float, h_err: float) -> dict:
        """获取浪形信息
        
        Args:
            shape: 浪形类型
            height: 浪高
            h_err: 水平偏差
        
        Returns:
            浪形信息字典
        """
        return {
            'wave_type': shape,
            'wave_height': height,
            'horizontal_error': h_err
        }
    
    def _update_history(self, result: Dict):
        """更新历史记录
        
        Args:
            result: 当前检测结果
        """
        self.history.append(result)
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def _calculate_optical_flow(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """计算光流
        
        Args:
            current_frame: 当前帧图像
        
        Returns:
            光流场
        """
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return None
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 使用Farneback光流算法
        flow = cv2.calcOpticalFlowFarneback(
            self.last_frame, current_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        self.last_frame = current_gray.copy()
        return flow
    
    def _smooth_detection(self, result: Dict) -> Dict:
        """平滑检测结果
        
        Args:
            result: 当前检测结果
        
        Returns:
            平滑后的检测结果
        """
        if len(self.history) < 3:
            return result
        
        # 平滑浪高
        if 'wave_height' in result:
            heights = [h.get('wave_height', 0) for h in self.history]
            heights.append(result['wave_height'])
            result['wave_height'] = round(np.mean(heights[-5:]), 2)
        
        # 平滑浪宽
        if 'wave_width' in result:
            widths = [w.get('wave_width', 0) for w in self.history]
            widths.append(result['wave_width'])
            result['wave_width'] = round(np.mean(widths[-5:]), 2)
        
        # 平滑位置
        if 'wave_position' in result and result['wave_position']:
            positions = [p.get('wave_position', {}) for p in self.history]
            positions.append(result['wave_position'])
            
            valid_positions = [p for p in positions if p]
            if len(valid_positions) > 2:
                avg_x = np.mean([p['x'] for p in valid_positions[-5:]])
                avg_y = np.mean([p['y'] for p in valid_positions[-5:]])
                result['wave_position'] = {
                    'x': round(avg_x, 2),
                    'y': round(avg_y, 2)
                }
        
        # 平滑浪形类型（使用多数投票）
        if 'wave_type' in result:
            types = [t.get('wave_type', '未知') for t in self.history]
            types.append(result['wave_type'])
            
            # 统计频率
            type_counts = {}
            for t in types[-8:]:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            # 选择最常见的类型
            if type_counts:
                most_common = max(type_counts, key=type_counts.get)
                result['wave_type'] = most_common
        
        return result
    
    def _temporal_optimization(self, result: Dict, frame: np.ndarray, info: Dict) -> Tuple[Dict, Dict]:
        """时序优化
        
        Args:
            result: 当前检测结果
            frame: 当前帧图像
            info: 检测信息
        
        Returns:
            优化后的检测结果和信息
        """
        # 计算光流
        flow = self._calculate_optical_flow(frame)
        if flow is not None:
            info['optical_flow_calculated'] = True
            # 可以使用光流信息来优化检测结果
            # 例如，预测浪形的运动方向和速度
        
        # 平滑检测结果
        result = self._smooth_detection(result)
        info['temporal_optimized'] = True
        
        # 前后帧对比，过滤异常值
        if len(self.history) > 0:
            previous = self.history[-1]
            # 检查浪高变化是否异常
            if 'wave_height' in result and 'wave_height' in previous:
                height_diff = abs(result['wave_height'] - previous['wave_height'])
                if height_diff > 3:  # 超过3mm的变化视为异常
                    # 使用历史平均值
                    result['wave_height'] = previous['wave_height']
                    info['height_anomaly_corrected'] = True
        
        return result, info
