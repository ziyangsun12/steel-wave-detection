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
    
    def __init__(self, config: Dict):
        """初始化检测参数
        
        Args:
            config: 检测配置参数
        """
        self.config = config
        self.detect_config = config.get('detection', {})
        
        # 浪形等级阈值
        self.wave_levels = self.detect_config.get('wave_levels', {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0
        })
        
        # 加载YOLOv8模型（如果使用）
        self.yolo_model = None
        # 这里可以加载预训练的YOLOv8模型
        if has_yolo:
            # self.yolo_model = YOLO('yolov8n-cls.pt')
            pass
    
    def detect(self, contour_data: Dict) -> Tuple[Dict, Dict]:
        """检测浪形
        
        Args:
            contour_data: 轮廓数据
        
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
