import cv2
import numpy as np
from collections import deque

class WaveDetector:
    def __init__(self):
        # 算法超参数配置
        self.gray_threshold = 150             
        self.wave_amplitude_threshold = 4.5   
        self.trim_ratio = 0.15                
        self.defog_strength = 5               
        
        # 历史振幅时序平滑队列 (FIFO)
        self.history_frames = 15  
        self.ws_amp_history = deque(maxlen=self.history_frames)
        self.ds_amp_history = deque(maxlen=self.history_frames)
        
        self.roi_locked = False
        self.roi_top = 0
        self.roi_bottom = 0
        self.current_perspective = "未知" 

    # ==========================================
    # 【第三部分】一维边缘信号处理：局部趋势跟踪削峰
    # 限制偏离局部基准线过大的极值尖刺，完美保留斜向带钢的倾斜趋势
    # ==========================================
    def despike_curve(self, curve, limit=35):
        if len(curve) < 31:
            return curve
        # 预先计算宏观局部基准线，替代全局中位数
        baseline = self.smooth_curve(curve, window_size=31)
        return np.clip(curve, baseline - limit, baseline + limit)

    # ==========================================
    # 【第三部分】一维边缘信号处理：一维卷积平滑滤波
    # 基于滑动窗口的低通滤波，消除物理像素网格的台阶状锯齿
    # ==========================================
    def smooth_curve(self, curve, window_size=41):
        window = np.ones(window_size) / window_size
        pad_len = window_size // 2
        padded = np.pad(curve, (pad_len, pad_len), mode='edge')
        smoothed = np.convolve(padded, window, mode='valid')
        if len(smoothed) > len(curve):
            smoothed = smoothed[:len(curve)]
        return smoothed

    # ==========================================
    # 【第三部分】一维边缘信号处理：一阶线性去趋势拉平与鲁棒极差统计
    # 彻底消除画面倾斜角与透视变形，提取纯粹的浪形振幅信号
    # ==========================================
    def calculate_true_amplitude(self, edge_curve, trim_len):
        if len(edge_curve) > 2 * trim_len and trim_len > 0:
            valid_curve = edge_curve[trim_len:-trim_len]
        else:
            valid_curve = edge_curve

        if len(valid_curve) < 10:
            return 0.0

        x = np.arange(len(valid_curve))
        # 采用最小二乘法进行一阶（Degree=1）线性回归直线拟合
        z = np.polyfit(x, valid_curve, 1)
        trend_line = np.polyval(z, x)
        
        # 曲线残差计算（实际坐标 - 拟合直线坐标）
        detrended = valid_curve - trend_line
        
        # 采用 90% 与 10% 分位数极差，实现抗极端异常值的鲁棒性测量
        amplitude = np.percentile(detrended, 90) - np.percentile(detrended, 10)
        return amplitude

    def process_frame(self, frame):
        h, w, _ = frame.shape
        heatmap = np.zeros((h, w, 3), dtype=np.uint8) 
        status = "未检测到带钢"
        annotated_frame = frame.copy()

        # ==========================================
        # 【第一部分】复杂工况下的带钢轮廓精准提取：光学特征防伪与隔离
        # ==========================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_gray = cv2.threshold(gray, self.gray_threshold, 255, cv2.THRESH_BINARY) 

        b, g, r = cv2.split(frame)
        red_dominance = cv2.subtract(r, b)
        
        # 特征 1：色彩空间差分提取（红热特征分离）
        _, mask_red = cv2.threshold(red_dominance, 15, 255, cv2.THRESH_BINARY)
        # 特征 2：绝对亮度补偿下限（极高光补偿机制，防止中心白热化断裂）
        _, mask_white_hot = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        
        # 逻辑掩码融合
        mask_valid_light = cv2.bitwise_or(mask_red, mask_white_hot)
        raw_mask = cv2.bitwise_and(mask_gray, mask_valid_light)

        # ==========================================
        # 【第一部分】复杂工况下的带钢轮廓精准提取：形态学去噪与粘连阻断
        # ==========================================
        kernel_size = max(3, int(self.defog_strength))
        
        # 动态生成特定尺寸的矩形结构元
        if self.current_perspective == "horizontal":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, kernel_size))
        elif self.current_perspective == "longitudinal":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, kernel_size + 5))

        # 形态学开运算：物理切断背景噪点与实体间的粘连
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        defogged_show = cv2.bitwise_and(frame, frame, mask=mask)

        # 渲染区域锁定标识
        if self.roi_locked and self.current_perspective == "horizontal":
            cv2.rectangle(annotated_frame, (0, self.roi_top), (w, self.roi_bottom), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "ROI LOCKED", (10, self.roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif self.roi_locked and self.current_perspective == "longitudinal":
            cv2.putText(annotated_frame, "ROI LOCKED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        depth_map = np.zeros((h, w), dtype=np.uint8)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            x, y, bw, bh = cv2.boundingRect(largest_contour)
            
            # 连通域拓扑分析：利用凸包实心度过滤破损区域
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            aspect_ratio = float(bw) / bh if bh > 0 else 0

            if area > 10000 and solidity > 0.50: 
                
                # ==========================================
                # 【第二部分】多视角的自适应匹配：几何特征嗅探与视角智能判断
                # ==========================================
                perspective = self.current_perspective
                
                # 根据长宽比 (Aspect Ratio) 进行分支选择
                if bw > w * 0.4 and aspect_ratio > 1.2:
                    perspective = "horizontal"
                elif bh > h * 0.4 and aspect_ratio <= 1.2:
                    perspective = "longitudinal"
                elif self.current_perspective == "未知":
                    if aspect_ratio > 1.2:
                        perspective = "horizontal"
                    else:
                        perspective = "longitudinal"

                # ==========================================
                # 【第二部分】多视角的自适应匹配：状态自纠正机制
                # 侦测到物理特征变化时，平滑重置状态机
                # ==========================================
                if perspective != "未知":
                    if not self.roi_locked or self.current_perspective != perspective:
                        margin = 30 
                        self.roi_top = max(0, y - margin)
                        self.roi_bottom = min(h, y + bh + margin)
                        self.roi_locked = True
                        self.current_perspective = perspective
                        self.ws_amp_history.clear()
                        self.ds_amp_history.clear()
                    
                    plate_mask = mask[y:y+bh, x:x+bw]
                    clean_mask = np.zeros_like(plate_mask)
                    simulated_depth = np.zeros_like(plate_mask, dtype=np.uint8)

                    # ------------------------------------------
                    # 模式 A：横向视角模型
                    # ------------------------------------------
                    if self.current_perspective == "horizontal":
                        valid_cols = np.any(plate_mask > 0, axis=0)
                        if np.sum(valid_cols) > 50: 
                            # 【第一部分】正交极值搜索：沿列方向获取上下边缘
                            top_edge_raw = np.argmax(plate_mask[:, valid_cols], axis=0)
                            bottom_edge_raw = bh - np.argmax(plate_mask[::-1, valid_cols], axis=0)
                            
                            top_edge_raw = self.despike_curve(top_edge_raw, limit=35)
                            bottom_edge_raw = self.despike_curve(bottom_edge_raw, limit=35)
                            top_edge_smooth = self.smooth_curve(top_edge_raw)
                            bottom_edge_smooth = self.smooth_curve(bottom_edge_raw)
                            
                            trim_px = int(len(top_edge_smooth) * self.trim_ratio)
                            raw_ws_amp = self.calculate_true_amplitude(top_edge_smooth, trim_px)
                            raw_ds_amp = self.calculate_true_amplitude(bottom_edge_smooth, trim_px)
                            
                            self.ws_amp_history.append(raw_ws_amp)
                            self.ds_amp_history.append(raw_ds_amp)
                            smooth_ws_amp = np.mean(self.ws_amp_history)
                            smooth_ds_amp = np.mean(self.ds_amp_history)
                            
                            if smooth_ws_amp > self.wave_amplitude_threshold and smooth_ds_amp > self.wave_amplitude_threshold:
                                status = "双边浪 (报警)"
                            elif smooth_ws_amp > self.wave_amplitude_threshold:
                                status = "WS侧单边浪 (报警)"
                            elif smooth_ds_amp > self.wave_amplitude_threshold:
                                status = "DS侧单边浪 (报警)"
                            else:
                                status = "平直"

                            valid_indices = np.where(valid_cols)[0]
                            for i, col in enumerate(valid_indices):
                                t = max(0, min(bh-1, int(top_edge_smooth[i])))
                                b = max(0, min(bh-1, int(bottom_edge_smooth[i]) - 8))
                                if b > t: 
                                    clean_mask[t:b, col] = 255 
                                    # 动态中轴距抛物线渲染模型
                                    center = (t + b) / 2.0
                                    half_w = (b - t) / 2.0
                                    y_coords = np.arange(t, b)
                                    dist = np.abs(y_coords - center) / max(1, half_w)
                                    intensities = 200 - 150 * (dist ** 2)
                                    simulated_depth[t:b, col] = np.clip(intensities, 50, 200)
                                    
                            draw_trim_px = int(bw * self.trim_ratio)
                            if draw_trim_px > 0:
                                line_x1, line_x2 = x + draw_trim_px, x + bw - draw_trim_px
                                cv2.line(annotated_frame, (line_x1, y), (line_x1, y + bh), (0, 255, 255), 2)
                                cv2.line(annotated_frame, (line_x2, y), (line_x2, y + bh), (0, 255, 255), 2)
                                cv2.putText(annotated_frame, f"CUT {int(self.trim_ratio*100)}%", (line_x1, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # ------------------------------------------
                    # 模式 B：纵向/斜向视角模型
                    # ------------------------------------------
                    elif self.current_perspective == "longitudinal":
                        valid_rows = np.any(plate_mask > 0, axis=1)
                        if np.sum(valid_rows) > 50:
                            # 【第一部分】正交极值搜索：沿行方向获取左右边缘
                            left_edge_raw = np.argmax(plate_mask[valid_rows, :], axis=1)
                            right_edge_raw = bw - np.argmax(plate_mask[valid_rows, ::-1], axis=1)
                            
                            left_edge_raw = self.despike_curve(left_edge_raw, limit=35)
                            right_edge_raw = self.despike_curve(right_edge_raw, limit=35)
                            left_edge_smooth = self.smooth_curve(left_edge_raw)
                            right_edge_smooth = self.smooth_curve(right_edge_raw)
                            
                            valid_indices = np.where(valid_rows)[0]
                            N = len(valid_indices)
                            
                            # ==========================================
                            # 【第二部分】多视角的自适应匹配：基于二维空间向量的正交斜向切割
                            # ==========================================
                            # 1. 计算带钢中心轴的倾斜方向单位向量 (ux, uy)
                            dx = ((right_edge_smooth[-1] + left_edge_smooth[-1]) / 2.0) - ((right_edge_smooth[0] + left_edge_smooth[0]) / 2.0)
                            dy = N - 1
                            L = np.hypot(dx, dy)
                            ux, uy = dx / L, dy / L

                            # 2. 向量点积投影：将二维坐标投射至一维绝对长度轴上
                            y_coords = np.arange(N)
                            p_center = ((left_edge_smooth + right_edge_smooth) / 2.0) * ux + y_coords * uy
                            p_min, p_max = p_center[0], p_center[-1]
                            P_len = p_max - p_min
                            
                            # 3. 确立正交切割截断区间
                            p_cut_low = p_min + self.trim_ratio * P_len
                            p_cut_high = p_max - self.trim_ratio * P_len
                            
                            p_left = left_edge_smooth * ux + y_coords * uy
                            mask_l = (p_left >= p_cut_low) & (p_left <= p_cut_high)
                            
                            p_right = right_edge_smooth * ux + y_coords * uy
                            mask_r = (p_right >= p_cut_low) & (p_right <= p_cut_high)
                            
                            left_trim = left_edge_smooth[np.where(mask_l)[0][0] : np.where(mask_l)[0][-1]+1] if np.any(mask_l) else left_edge_smooth
                            right_trim = right_edge_smooth[np.where(mask_r)[0][0] : np.where(mask_r)[0][-1]+1] if np.any(mask_r) else right_edge_smooth
                            
                            raw_left_amp = self.calculate_true_amplitude(left_trim, 0)
                            raw_right_amp = self.calculate_true_amplitude(right_trim, 0)
                            
                            self.ds_amp_history.append(raw_left_amp)
                            self.ws_amp_history.append(raw_right_amp)
                            smooth_ds_amp = np.mean(self.ds_amp_history)
                            smooth_ws_amp = np.mean(self.ws_amp_history)
                            
                            if smooth_ws_amp > self.wave_amplitude_threshold and smooth_ds_amp > self.wave_amplitude_threshold:
                                status = "双边浪 (报警)"
                            elif smooth_ws_amp > self.wave_amplitude_threshold:
                                status = "WS侧单边浪 (报警)"
                            elif smooth_ds_amp > self.wave_amplitude_threshold:
                                status = "DS侧单边浪 (报警)"
                            else:
                                status = "平直"

                            for i, row in enumerate(valid_indices):
                                l = max(0, min(bw-1, int(left_edge_smooth[i])))
                                r = max(0, min(bw-1, int(right_edge_smooth[i])))
                                if r > l: 
                                    clean_mask[row, l:r] = 255
                                    # 动态中轴距抛物线渲染模型，跟随带钢倾角自适应旋转
                                    center = (l + r) / 2.0
                                    half_w = (r - l) / 2.0
                                    x_coords_arr = np.arange(l, r)
                                    dist = np.abs(x_coords_arr - center) / max(1, half_w)
                                    intensities = 200 - 150 * (dist ** 2)
                                    simulated_depth[row, l:r] = np.clip(intensities, 50, 200)
                                    
                            # 反推求解斜向切割基准线方程并渲染
                            if self.trim_ratio > 0:
                                y_start = y + valid_indices[0]
                                
                                Y1_low = (p_cut_low - 0 * ux) / uy
                                Y2_low = (p_cut_low - bw * ux) / uy
                                pt1_low = (int(x), int(y_start + Y1_low))
                                pt2_low = (int(x + bw), int(y_start + Y2_low))
                                cv2.line(annotated_frame, pt1_low, pt2_low, (0, 255, 255), 2)
                                cv2.putText(annotated_frame, f"CUT {int(self.trim_ratio*100)}%", (pt1_low[0]+10, pt1_low[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                                Y1_high = (p_cut_high - 0 * ux) / uy
                                Y2_high = (p_cut_high - bw * ux) / uy
                                pt1_high = (int(x), int(y_start + Y1_high))
                                pt2_high = (int(x + bw), int(y_start + Y2_high))
                                cv2.line(annotated_frame, pt1_high, pt2_high, (0, 255, 255), 2)
                                cv2.putText(annotated_frame, f"CUT {int(self.trim_ratio*100)}%", (pt1_high[0]+10, pt1_high[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    depth_map[y:y+bh, x:x+bw] = cv2.bitwise_and(simulated_depth, simulated_depth, mask=clean_mask)
                    c_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if c_contours:
                        shifted_contours = [c + np.array([x, y]) for c in c_contours]
                        cv2.drawContours(annotated_frame, shifted_contours, -1, (0, 255, 0), 2)
                else:
                    status = "未检测到带钢"
            else:
                status = "未检测到带钢"
        else:
            self.roi_locked = False
            self.current_perspective = "未知" 
            self.ws_amp_history.clear()
            self.ds_amp_history.clear()

        if status != "未检测到带钢":
            heatmap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            heatmap[depth_map == 0] = [0, 0, 0] 

        return heatmap, status, annotated_frame, defogged_show