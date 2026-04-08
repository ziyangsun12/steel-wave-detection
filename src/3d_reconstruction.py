"""带钢三维轮廓重建模块"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class Reconstructor:
    """带钢三维轮廓重建类"""
    
    def __init__(self, config: Dict):
        """初始化重建参数
        
        Args:
            config: 重建配置参数
        """
        self.config = config
        self.reconstruct_config = config.get('reconstruction', {})
        self.calib_config = config.get('calibration', {})
        
        # 像素到毫米的转换系数
        self.pixel_to_mm = self.calib_config.get('pixel_to_mm', {'x': 0.1, 'y': 0.11})
        
    def reconstruct(self, binary: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Dict, Dict]:
        """重建三维轮廓
        
        Args:
            binary: 二值图像
            mask: 带钢掩码
        
        Returns:
            三维轮廓数据和重建信息
        """
        info = {}
        
        # 检查输入
        if binary is None or binary.size == 0:
            info['error'] = '无效输入'
            return {}, info
        
        # 提取带钢边界
        edges = self._extract_edges(binary, mask)
        if not edges:
            info['no_edges'] = True
            return {}, info
        
        # 提取中线和边线
        center_line, left_edge, right_edge = self._extract_lines(edges)
        info['lines_extracted'] = True
        
        # 计算高度场
        height_field = self._calculate_height_field(center_line, left_edge, right_edge)
        info['height_field_calculated'] = True
        
        # 平面拟合
        if self.reconstruct_config.get('plane_fitting', {}).get('enable', True):
            plane_params, fit_info = self._fit_plane(height_field)
            info.update(fit_info)
        else:
            plane_params = None
        
        # 计算高度差
        height_diff = self._calculate_height_diff(height_field, plane_params)
        info['height_diff_calculated'] = True
        
        # 构建轮廓数据
        contour_data = {
            'center_line': center_line,
            'left_edge': left_edge,
            'right_edge': right_edge,
            'height_field': height_field,
            'height_diff': height_diff,
            'plane_params': plane_params
        }
        
        return contour_data, info
    
    def _extract_edges(self, binary: np.ndarray, mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """提取带钢边界
        
        Args:
            binary: 二值图像
            mask: 带钢掩码
        
        Returns:
            边界轮廓列表
        """
        # 应用掩码
        if mask is not None:
            binary = cv2.bitwise_and(binary, mask)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        min_area = self.config.get('exception', {}).get('no_steel', {}).get('min_contour_area', 10000)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return valid_contours
    
    def _extract_lines(self, edges: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取中线和边线
        
        Args:
            edges: 边界轮廓列表
        
        Returns:
            中线、左边线、右边线
        """
        if not edges:
            return np.array([]), np.array([]), np.array([])
        
        # 找到最大轮廓
        max_contour = max(edges, key=cv2.contourArea)
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 提取左边线和右边线
        left_edge = []
        right_edge = []
        
        for y_coord in range(y, y + h):
            # 找到当前行的左右边界点
            row = max_contour[max_contour[:, :, 1] == y_coord]
            if len(row) > 0:
                row = row[:, 0, :]
                left = row[np.argmin(row[:, 0])]
                right = row[np.argmax(row[:, 0])]
                left_edge.append(left)
                right_edge.append(right)
        
        left_edge = np.array(left_edge)
        right_edge = np.array(right_edge)
        
        # 计算中线
        center_line = (left_edge + right_edge) // 2
        
        return center_line, left_edge, right_edge
    
    def _calculate_height_field(self, center_line: np.ndarray, left_edge: np.ndarray, right_edge: np.ndarray) -> np.ndarray:
        """计算高度场
        
        Args:
            center_line: 中线点集
            left_edge: 左边线点集
            right_edge: 右边线点集
        
        Returns:
            高度场数据
        """
        if len(center_line) == 0:
            return np.array([])
        
        # 计算宽度变化作为高度的代理
        width = np.linalg.norm(right_edge - left_edge, axis=1)
        
        # 转换为毫米
        width_mm = width * self.pixel_to_mm['x']
        
        # 计算高度场（这里使用宽度变化作为高度的近似）
        # 实际应用中可能需要更复杂的算法，如结构光或立体视觉
        height_field = np.zeros((len(center_line), 3))
        height_field[:, 0] = center_line[:, 1] * self.pixel_to_mm['y']  # y坐标（纵向）
        height_field[:, 1] = center_line[:, 0] * self.pixel_to_mm['x']  # x坐标（横向）
        height_field[:, 2] = width_mm  # 高度（使用宽度变化）
        
        return height_field
    
    def _fit_plane(self, height_field: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """拟合平面
        
        Args:
            height_field: 高度场数据
        
        Returns:
            平面参数和拟合信息
        """
        info = {'plane_fitted': True}
        
        if len(height_field) < 3:
            info['error'] = '点数不足'
            return None, info
        
        try:
            # 提取坐标
            x = height_field[:, 0]
            y = height_field[:, 1]
            z = height_field[:, 2]
            
            # 构建设计矩阵
            A = np.vstack([x, y, np.ones(len(x))]).T
            
            # 最小二乘拟合
            plane_params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            
            # 计算拟合误差
            z_pred = plane_params[0] * x + plane_params[1] * y + plane_params[2]
            error = np.mean(np.abs(z - z_pred))
            info['fit_error'] = error
            
            return plane_params, info
        except Exception as e:
            info['error'] = str(e)
            return None, info
    
    def _calculate_height_diff(self, height_field: np.ndarray, plane_params: Optional[np.ndarray]) -> np.ndarray:
        """计算高度差
        
        Args:
            height_field: 高度场数据
            plane_params: 平面参数
        
        Returns:
            高度差数据
        """
        if len(height_field) == 0:
            return np.array([])
        
        if plane_params is None:
            # 如果没有拟合平面，使用平均高度作为参考
            mean_height = np.mean(height_field[:, 2])
            height_diff = height_field[:, 2] - mean_height
        else:
            # 使用拟合平面作为参考
            x = height_field[:, 0]
            y = height_field[:, 1]
            z_pred = plane_params[0] * x + plane_params[1] * y + plane_params[2]
            height_diff = height_field[:, 2] - z_pred
        
        return height_diff
    
    def calculate_wave_parameters(self, contour_data: Dict) -> Dict:
        """计算浪形参数
        
        Args:
            contour_data: 轮廓数据
        
        Returns:
            浪形参数
        """
        params = {}
        
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
                peak_positions = np.array(peaks) * self.pixel_to_mm['y']
                wave_widths = np.diff(peak_positions)
                avg_wave_width = np.mean(wave_widths)
                params['wave_width'] = round(avg_wave_width, 2)
            
            # 计算浪形位置
            if peaks:
                max_peak_idx = peaks[np.argmax(height_diff[peaks])]
                max_peak_pos = center_line[max_peak_idx]
                params['wave_position'] = {
                    'x': round(max_peak_pos[0] * self.pixel_to_mm['x'], 2),
                    'y': round(max_peak_pos[1] * self.pixel_to_mm['y'], 2)
                }
        
        return params
    
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
    
    def get_3d_points(self, contour_data: Dict) -> np.ndarray:
        """获取三维点云
        
        Args:
            contour_data: 轮廓数据
        
        Returns:
            三维点云数据
        """
        height_field = contour_data.get('height_field', [])
        if len(height_field) == 0:
            return np.array([])
        
        # 转换为三维点云
        points_3d = np.zeros((len(height_field), 3))
        points_3d[:, 0] = height_field[:, 1]  # x坐标
        points_3d[:, 1] = height_field[:, 0]  # y坐标
        points_3d[:, 2] = height_field[:, 2]  # z坐标（高度）
        
        return points_3d
