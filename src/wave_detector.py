import cv2
import numpy as np
from collections import deque

class WaveDetector:
    def __init__(self):
        # 默认参数
        self.omega = 0.8
        self.t0 = 0.25
        self.binary_threshold = 120
        
        # ==========================================
        # 模块 1：系统核心参数与状态机初始化
        # ==========================================
        
        # 1. 浪形振幅报警阈值：
        # 超过此值（像素级的高度差）即判定为浪形。配合时间平滑算法，4.5 是一个兼顾灵敏与稳定的值。
        self.wave_amplitude_threshold = 4.5  
        
        # 2. 滑动窗口（时间平滑）系统：用于消除单帧画面的闪烁与误报
        self.history_frames = 15  # 记录最近 15 帧的数据 (在 30fps 下约等于 0.5 秒)
        # deque(双端队列)：当存满 15 个数据后，新数据进入会自动挤出最老的数据，非常适合做滑动平均
        self.ws_amp_history = deque(maxlen=self.history_frames) # 操作侧 (上方) 振幅历史
        self.ds_amp_history = deque(maxlen=self.history_frames) # 传动侧 (下方) 振幅历史
        
        # 3. 动态 ROI (感兴趣区域) 与状态记录器
        self.roi_locked = False   # 标志位：当前是否已经锁定了带钢所在的纵向区域
        self.roi_top = 0          # 锁定区域的上边界 Y 坐标
        self.roi_bottom = 0       # 锁定区域的下边界 Y 坐标
        self.stable_frames = 0    # 记录带钢在画面中稳定存在了多少帧（用于头尾过滤）
        
        # 4. 像素到毫米的转换系数（根据实际相机参数和距离调整）
        self.pixel_to_mm = 0.5  # 假设 1 像素 = 0.5 毫米
        
        # 5. 浪形等级阈值
        self.wave_levels = {
            "低": 4.5,
            "中": 8.0,
            "高": 12.0
        }
    
    def set_parameters(self, omega, t0, binary_threshold):
        """设置参数"""
        self.omega = omega
        self.t0 = t0
        self.binary_threshold = binary_threshold

    # ==========================================
    # 模块 2：核心数学信号处理工具
    # ==========================================
    
    def dark_channel_prior(self, img, size=15):
        """
        暗通道去雾算法
        """
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def dehaze(self, img):
        """
        图像去雾
        """
        img = img.astype(np.float64) / 255.0
        dark_channel = self.dark_channel_prior(img)
        atmospheric_light = np.max(img, axis=(0, 1))
        transmission = 1 - self.omega * dark_channel / np.max(dark_channel)
        transmission = np.maximum(transmission, self.t0)
        
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
        
        result = np.clip(result, 0, 1) * 255
        return result.astype(np.uint8)
    
    def enhance_contrast(self, img):
        """
        CLAHE 对比度增强
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_img
    
    def smooth_curve(self, curve, window_size=21):
        """
        步骤 A：一维信号均值滤波。
        目的：消除图像形态学处理（开运算）带来的边缘“像素级阶梯/锯齿”，还原带钢真实的平滑轮廓。
        """
        window = np.ones(window_size) / window_size
        pad_len = window_size // 2
        # 使用 edge 模式进行边缘填充，防止卷积后曲线首尾出现断崖式塌陷
        padded = np.pad(curve, (pad_len, pad_len), mode='edge')
        smoothed = np.convolve(padded, window, mode='valid')
        if len(smoothed) > len(curve):
            smoothed = smoothed[:len(curve)] # 确保输出与输入长度严格对齐
        return smoothed

    def calculate_true_amplitude(self, edge_curve, trim_ratio=0.15):
        """
        步骤 B：真实振幅计算（抗畸变、抗斜边算法）。
        目的：剥离摄像头透视畸变和带钢自然下垂的影响，只提取纯粹的浪形起伏。
        返回：振幅、浪高、浪宽
        """
        # 1. 掐头去尾：裁掉左右两端各 15% 的斜边区域，只对带钢中间最核心的 70% 区域进行运算
        trim_len = int(len(edge_curve) * trim_ratio)
        if len(edge_curve) > 2 * trim_len and trim_len > 0:
            valid_curve = edge_curve[trim_len:-trim_len]
        else:
            valid_curve = edge_curve

        if len(valid_curve) < 10:
            return 0.0, 0.0, 0.0

        x = np.arange(len(valid_curve))
        
        # 2. 二阶抛物线拟合：找出带钢边缘整体的“趋势线”（哪怕它倾斜或者微微下垂）
        z = np.polyfit(x, valid_curve, 2)
        trend_curve = np.polyval(z, x)
        
        # 3. 信号去趋势 (Detrending)：实际轮廓 减去 趋势线，将带钢绝对“拉平”
        detrended = valid_curve - trend_curve
        
        # 4. 计算极差：使用 90% 分位数减去 10% 分位数，彻底无视极端毛刺噪点
        amplitude = np.percentile(detrended, 90) - np.percentile(detrended, 10)
        
        # 5. 计算浪高（最大峰值与最小谷值之差）
        wave_height = np.max(detrended) - np.min(detrended)
        
        # 6. 计算浪宽（通过寻找峰值和谷值的位置）
        wave_width = 0
        if len(detrended) > 20:
            # 使用一阶导数找极值点
            diff = np.diff(detrended)
            extrema = np.where(np.diff(np.sign(diff)))[0]
            if len(extrema) >= 2:
                # 计算相邻极值点之间的距离
                widths = np.diff(extrema)
                wave_width = np.mean(widths) if len(widths) > 0 else 0
        
        return amplitude, wave_height, wave_width

    # ==========================================
    # 模块 3：主视频帧处理流水线 (每来一帧图执行一次)
    # ==========================================

    def process_frame(self, frame):
        import time
        start_time = time.time()
        h, w, _ = frame.shape
        heatmap = np.zeros((h, w, 3), dtype=np.uint8) # 初始化空的热力图
        status = "未检测到带钢"                       # 初始化默认状态
        algorithm_time = 0
        dehazed_frame = frame.copy()  # 初始化去雾后的帧
        contour_frame = frame.copy()  # 初始化轮廓提取后的帧
        wave_height = 0.0  # 浪高
        wave_width = 0.0   # 浪宽
        wave_level = "无"  # 浪形等级

        # --- 第一步：图像预处理与自适应二值化 ---
        # 1. 暗通道去雾
        dehazed_frame = self.dehaze(frame)
        # 2. CLAHE 对比度增强
        dehazed_frame = self.enhance_contrast(dehazed_frame)
        # 3. 使用 HSV 色彩空间，提取 V 通道并使用 Otsu 自适应阈值
        hsv = cv2.cvtColor(dehazed_frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        # 4. 双边滤波：平滑噪声的同时保留边缘
        v_channel = cv2.bilateralFilter(v_channel, 9, 75, 75)
        # 5. 二值化 - 提高阈值以突出亮白红色的带钢区域
        _, mask = cv2.threshold(v_channel, self.binary_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

        # --- 第二步：物理空间隔离与形态学去雾 ---
        if self.roi_locked:
            # 如果已经锁定了带钢轨道：暴力抹黑轨道上下的所有区域，彻底物理隔绝大面积水雾
            mask[0:self.roi_top, :] = 0
            mask[self.roi_bottom:h, :] = 0
            # 使用宽扁核(15, 5)切断带钢下边缘与辊道的垂直反光粘连
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        else:
            # 没锁定前：使用大尺寸核(25, 15)强力去除全屏的丝状水雾，以便寻找第一块主钢板
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
            
        # 形态学开运算：先腐蚀（消灭细小噪点），再膨胀（恢复主体形状）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # 形态学闭运算：填充内部孔洞
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # --- 第三步：轮廓提取与目标确认 ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        depth_map = np.zeros((h, w), dtype=np.uint8) # 初始化空的高度图矩阵

        if contours:
            # 找到画面中面积最大的连通域
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            x, y, bw, bh = cv2.boundingRect(largest_contour)
            
            # 计算轮廓实心度（面积与凸包面积的比例）
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # 条件确认：面积够大 (>15000) 且 宽度占画面的 30% 以上，且实心度大于 0.8，才确认为真正的带钢
            if area > 15000 and bw > w * 0.3 and solidity > 0.8: 
                
                # --- 第四步：触发 ROI 锁定系统 ---
                if not self.roi_locked:
                    margin = 30 # 上下预留 30 像素的容错空间
                    self.roi_top = max(0, y - margin)
                    self.roi_bottom = min(h, y + bh + margin)
                    self.roi_locked = True
                    self.stable_frames = 0 # 新带钢咬钢，重置稳定计数
                    # 清空上一根带钢遗留的历史振幅数据
                    self.ws_amp_history.clear()
                    self.ds_amp_history.clear()
                
                self.stable_frames += 1 # 稳定帧数递增
                
                # --- 第五步：提取带钢精确边缘曲线 ---
                plate_mask = mask[y:y+bh, x:x+bw]
                valid_cols = np.any(plate_mask > 0, axis=0) # 找出有非黑像素的有效列
                
                if np.sum(valid_cols) > 50: # 有效列足够多才进行计算
                    # argmax 寻找每列第一个非零像素点，获取最原始的上下边缘坐标
                    top_edge_raw = np.argmax(plate_mask[:, valid_cols], axis=0)
                    bottom_edge_raw = bh - np.argmax(plate_mask[::-1, valid_cols], axis=0)
                    
                    # 调用工具函数，对原始边缘进行一维滤波磨皮
                    top_edge_smooth = self.smooth_curve(top_edge_raw, window_size=21)
                    bottom_edge_smooth = self.smooth_curve(bottom_edge_raw, window_size=21)
                    
                    # --- 第六步：过渡区延时启动与状态判定 ---
                    # 设定 30 帧 (约 1 秒) 作为缓冲期。等待带钢彻底进入画面且水雾散开
                    is_stable_body = (self.stable_frames > 30)
                    
                    if is_stable_body:
                        # 1. 计算当前单帧的真实振幅（已去除畸变）
                        raw_ws_amp, ws_wave_height, ws_wave_width = self.calculate_true_amplitude(top_edge_smooth)
                        raw_ds_amp, ds_wave_height, ds_wave_width = self.calculate_true_amplitude(bottom_edge_smooth)
                        
                        # 2. 计算实际浪高浪宽（转换为毫米）
                        wave_height = max(ws_wave_height, ds_wave_height) * self.pixel_to_mm
                        wave_width = max(ws_wave_width, ds_wave_width) * self.pixel_to_mm
                        
                        # 3. 确定浪形等级
                        if wave_height < self.wave_levels["低"]:
                            wave_level = "无"
                        elif wave_height < self.wave_levels["中"]:
                            wave_level = "低"
                        elif wave_height < self.wave_levels["高"]:
                            wave_level = "中"
                        else:
                            wave_level = "高"
                        
                        # 4. 存入历史队列
                        self.ws_amp_history.append(raw_ws_amp)
                        self.ds_amp_history.append(raw_ds_amp)
                        
                        # 5. 计算最近一段时间的平均振幅（防闪烁）
                        smooth_ws_amp = np.mean(self.ws_amp_history)
                        smooth_ds_amp = np.mean(self.ds_amp_history)
                        
                        # 6. 根据平滑后的振幅与阈值比对，输出最终的工业报警信号
                        if smooth_ws_amp > self.wave_amplitude_threshold and smooth_ds_amp > self.wave_amplitude_threshold:
                            status = f"双边浪 (报警)"
                        elif smooth_ws_amp > self.wave_amplitude_threshold:
                            status = f"WS侧单边浪 (报警)"
                        elif smooth_ds_amp > self.wave_amplitude_threshold:
                            status = f"DS侧单边浪 (报警)"
                        else:
                            status = "平直"
                    else:
                        # 在刚咬钢的 1 秒内，强制休眠检测逻辑，输出过渡区
                        status = "过渡区"

                    # --- 第七步：3D 深度贴图重构 (Clean Mask 重绘) ---
                    # 彻底丢弃带有毛刺的原始轮廓，用完美的平滑曲线在黑板上重新“画”出一个带钢
                    clean_mask = np.zeros_like(plate_mask)
                    valid_indices = np.where(valid_cols)[0]
                    
                    for i, col in enumerate(valid_indices):
                        t = int(top_edge_smooth[i])
                        # 底部往上强行提 8 个像素，彻底切掉下巴上可能的传送带残影
                        b = int(bottom_edge_smooth[i]) - 8 
                        t = max(0, min(bh-1, t)) # 越界保护
                        b = max(0, min(bh-1, b))
                        if b > t:
                            clean_mask[t:b, col] = 255 # 填充白色实心
                            
                    # 构造一个立体的纵向高度渐变（让图像有 3D 圆柱表面的隆起感）
                    gradient = np.linspace(50, 200, bh, dtype=np.float32)
                    gradient_2d = np.tile(gradient, (bw, 1)).T
                    simulated_depth = np.clip(gradient_2d, 0, 255).astype(np.uint8)
                    
                    # 将深度贴图精确地覆盖到我们刚刚画好的 Clean Mask 上
                    depth_map[y:y+bh, x:x+bw] = cv2.bitwise_and(simulated_depth, simulated_depth, mask=clean_mask)
                    
                else:
                    status = "未检测到带钢"
        else:
            # --- 第八步：抛钢复位系统 ---
            # 当带钢完全离开画面（没有巨大的亮斑），解除 ROI 锁定，清空所有历史数据
            self.roi_locked = False
            self.stable_frames = 0
            self.ws_amp_history.clear()
            self.ds_amp_history.clear()

        # 计算算法处理时间
        algorithm_time = (time.time() - start_time) * 1000

        # 生成轮廓提取后的帧
        contour_frame = dehazed_frame.copy()
        if contours:
            # 在轮廓提取后的帧上绘制轮廓
            cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
            # 如果有最大轮廓，绘制其边界框
            if 'largest_contour' in locals():
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                cv2.rectangle(contour_frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        # --- 第九步：伪彩色热力图渲染 ---
        if status != "未检测到带钢":
            # 将灰度的深度贴图，转换为 Jet 伪彩色（红高蓝低），极具工业检测仪表的质感
            heatmap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            # 将没有钢板的背景区域强制设回纯黑
            heatmap[depth_map == 0] = [0, 0, 0] 

        # 保存边缘数据
        edge_data = {
            'top_edge_raw': top_edge_raw.tolist() if 'top_edge_raw' in locals() else [],
            'bottom_edge_raw': bottom_edge_raw.tolist() if 'bottom_edge_raw' in locals() else [],
            'top_edge_smooth': top_edge_smooth.tolist() if 'top_edge_smooth' in locals() else [],
            'bottom_edge_smooth': bottom_edge_smooth.tolist() if 'bottom_edge_smooth' in locals() else [],
            'valid_cols': valid_cols.tolist() if 'valid_cols' in locals() else []
        }
        
        # 返回渲染好的图像、检测状态、处理时间、去雾后的帧和轮廓提取后的帧，交由前端 UI 进行显示
        return heatmap, status, algorithm_time, dehazed_frame, contour_frame, wave_height, wave_width, wave_level, edge_data
    
    def detect(self, contour_data, frame):
        """检测浪形
        
        Args:
            contour_data: 轮廓数据
            frame: 输入帧图像
        
        Returns:
            (检测结果, 检测信息)
        """
        # 调用process_frame处理帧
        heatmap, status, algorithm_time, dehazed_frame, contour_frame, wave_height, wave_width, wave_level, edge_data = self.process_frame(frame)
        
        # 构建检测结果
        detection_result = {
            'wave_type': status,
            'wave_level': wave_level,
            'wave_height': wave_height,
            'wave_width': wave_width,
            'wave_position': {'x': 0, 'y': 0}
        }
        
        # 构建检测信息
        detect_info = {
            'algorithm_time': algorithm_time,
            'status': status,
            'wave_height': wave_height,
            'wave_width': wave_width,
            'wave_level': wave_level,
            'edge_data': edge_data
        }
        
        return detection_result, detect_info