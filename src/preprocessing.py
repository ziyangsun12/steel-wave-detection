"""带钢图像预处理模块"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class Preprocessor:
    """带钢图像预处理类"""
    
    def __init__(self, config: Dict):
        """初始化预处理参数
        
        Args:
            config: 预处理配置参数
        """
        self.config = config
        self.preprocess_config = config.get('preprocessing', {})
        
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """预处理图像
        
        Args:
            frame: 输入帧图像
        
        Returns:
            预处理后的图像和处理信息
        """
        info = {}
        
        # 检查帧是否有效
        if frame is None or frame.size == 0:
            info['error'] = '无效帧'
            return frame, info
        
        # 去雾处理
        if self.preprocess_config.get('dehaze', {}).get('enable', False):
            frame, dehaze_info = self._dehaze(frame)
            info.update(dehaze_info)
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info['gray_shape'] = gray.shape
        
        # 去噪处理
        if self.preprocess_config.get('denoise', {}).get('enable', False):
            gray, denoise_info = self._denoise(gray)
            info.update(denoise_info)
        
        # 对比度增强
        if self.preprocess_config.get('contrast', {}).get('enable', False):
            gray, contrast_info = self._enhance_contrast(gray)
            info.update(contrast_info)
        
        # 阈值处理
        binary, threshold_info = self._threshold(gray)
        info.update(threshold_info)
        
        # 形态学操作
        if self.preprocess_config.get('morphology', {}).get('enable', False):
            binary, morphology_info = self._morphology(binary)
            info.update(morphology_info)
        
        # 异常检测
        info.update(self._detect_anomalies(frame, gray))
        
        return binary, info
    
    def _dehaze(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """去雾处理
        
        Args:
            frame: 输入帧图像
        
        Returns:
            去雾后的图像和处理信息
        """
        info = {'dehaze_applied': True}
        
        # 使用暗通道先验去雾
        if self.preprocess_config.get('dehaze', {}).get('dark_channel_prior', True):
            try:
                dehazed = self._dark_channel_prior(frame)
                info['dehaze_method'] = 'dark_channel_prior'
                return dehazed, info
            except Exception as e:
                info['dehaze_error'] = str(e)
                return frame, info
        
        return frame, info
    
    def _dark_channel_prior(self, frame: np.ndarray, omega: float = 0.95, window_size: int = 15) -> np.ndarray:
        """暗通道先验去雾算法
        
        Args:
            frame: 输入帧图像
            omega: 透射率估计参数
            window_size: 窗口大小
        
        Returns:
            去雾后的图像
        """
        # 转换为浮点数
        img = frame.astype(np.float64) / 255.0
        
        # 计算暗通道
        min_channel = np.min(img, axis=2)
        dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size)))
        
        # 估计大气光
        atmospheric_light = np.max(img[dark_channel > np.percentile(dark_channel, 95)])
        
        # 估计透射率
        transmission = 1 - omega * dark_channel / atmospheric_light
        transmission = np.maximum(transmission, 0.1)  # 防止透射率过小
        
        # 恢复图像
        dehazed = np.zeros_like(img)
        for i in range(3):
            dehazed[:, :, i] = (img[:, :, i] - atmospheric_light) / transmission + atmospheric_light
        
        # 裁剪到有效范围
        dehazed = np.clip(dehazed, 0, 1)
        dehazed = (dehazed * 255).astype(np.uint8)
        
        return dehazed
    
    def _denoise(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """去噪处理
        
        Args:
            gray: 灰度图像
        
        Returns:
            去噪后的图像和处理信息
        """
        info = {'denoise_applied': True}
        
        method = self.preprocess_config.get('denoise', {}).get('method', 'gaussian')
        kernel_size = self.preprocess_config.get('denoise', {}).get('kernel_size', 7)
        
        if method == 'gaussian':
            denoised = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            info['denoise_method'] = 'gaussian'
        elif method == 'median':
            denoised = cv2.medianBlur(gray, kernel_size)
            info['denoise_method'] = 'median'
        elif method == 'bilateral':
            denoised = cv2.bilateralFilter(gray, kernel_size, 75, 75)
            info['denoise_method'] = 'bilateral'
        else:
            denoised = gray
            info['denoise_method'] = 'none'
        
        return denoised, info
    
    def _enhance_contrast(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """增强对比度
        
        Args:
            gray: 灰度图像
        
        Returns:
            增强后的图像和处理信息
        """
        info = {'contrast_enhanced': True}
        
        method = self.preprocess_config.get('contrast', {}).get('method', 'clahe')
        
        if method == 'clahe':
            clip_limit = self.preprocess_config.get('contrast', {}).get('clip_limit', 2.0)
            tile_grid_size = tuple(self.preprocess_config.get('contrast', {}).get('tile_grid_size', [8, 8]))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(gray)
            info['contrast_method'] = 'clahe'
        elif method == 'histogram':
            enhanced = cv2.equalizeHist(gray)
            info['contrast_method'] = 'histogram'
        else:
            enhanced = gray
            info['contrast_method'] = 'none'
        
        return enhanced, info
    
    def _threshold(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """阈值处理
        
        Args:
            gray: 灰度图像
        
        Returns:
            二值图像和处理信息
        """
        info = {'threshold_applied': True}
        
        method = self.preprocess_config.get('threshold', {}).get('method', 'adaptive')
        
        if method == 'adaptive':
            block_size = self.preprocess_config.get('threshold', {}).get('block_size', 15)
            c = self.preprocess_config.get('threshold', {}).get('c', 10)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, block_size, c
            )
            info['threshold_method'] = 'adaptive'
        elif method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            info['threshold_method'] = 'otsu'
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            info['threshold_method'] = 'global'
        
        return binary, info
    
    def _morphology(self, binary: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """形态学操作
        
        Args:
            binary: 二值图像
        
        Returns:
            处理后的图像和处理信息
        """
        info = {'morphology_applied': True}
        
        kernel_size = tuple(self.preprocess_config.get('morphology', {}).get('kernel_size', [5, 5]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        operations = self.preprocess_config.get('morphology', {}).get('operations', ['close', 'open'])
        
        for op in operations:
            if op == 'close':
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            elif op == 'open':
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            elif op == 'dilate':
                binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
            elif op == 'erode':
                binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
        
        info['morphology_operations'] = operations
        return binary, info
    
    def _detect_anomalies(self, frame: np.ndarray, gray: np.ndarray) -> Dict:
        """检测异常情况
        
        Args:
            frame: 原始帧图像
            gray: 灰度图像
        
        Returns:
            异常检测信息
        """
        info = {}
        
        # 检测过曝
        if self.config.get('exception', {}).get('overexposure', {}).get('enable', True):
            threshold = self.config.get('exception', {}).get('overexposure', {}).get('threshold', 240)
            overexposed_pixels = np.sum(gray > threshold)
            overexposure_ratio = overexposed_pixels / gray.size
            info['overexposure_ratio'] = overexposure_ratio
            info['is_overexposed'] = overexposure_ratio > 0.5
        
        # 检测黑屏
        if self.config.get('exception', {}).get('black_screen', {}).get('enable', True):
            threshold = self.config.get('exception', {}).get('black_screen', {}).get('threshold', 10)
            dark_pixels = np.sum(gray < threshold)
            black_ratio = dark_pixels / gray.size
            info['black_ratio'] = black_ratio
            info['is_black_screen'] = black_ratio > 0.95
        
        return info
    
    def segment_steel(self, binary: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """分割带钢区域
        
        Args:
            binary: 二值图像
        
        Returns:
            带钢掩码和分割信息
        """
        info = {'segmentation_applied': True}
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            info['no_contours'] = True
            return None, info
        
        # 找到最大轮廓（带钢）
        max_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(max_contour)
        info['contour_area'] = contour_area
        
        # 检查是否有带钢
        min_area = self.config.get('exception', {}).get('no_steel', {}).get('min_contour_area', 10000)
        if contour_area < min_area:
            info['no_steel'] = True
            return None, info
        
        # 创建掩码
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        
        # 计算带钢边界
        x, y, w, h = cv2.boundingRect(max_contour)
        info['steel_bbox'] = (x, y, w, h)
        info['steel_width'] = w
        info['steel_height'] = h
        
        return mask, info
