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
        self.ws_amp_history = deque(maxlen=self.history_frames)  # 操作侧 (上方) 振幅历史
        self.ds_amp_history = deque(maxlen=self.history_frames)  # 传动侧 (下方) 振幅历史

        # 3. 动态 ROI (感兴趣区域) 与状态记录器
        self.roi_locked = False  # 标志位：当前是否已经锁定了带钢所在的纵向区域
        self.roi_top = 0  # 锁定区域的上边界 Y 坐标
        self.roi_bottom = 0  # 锁定区域的下边界 Y 坐标
        self.stable_frames = 0  # 记录带钢在画面中稳定存在了多少帧（用于头尾过滤）

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
            smoothed = smoothed[:len(curve)]  # 确保输出与输入长度严格对齐
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
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)  # 初始化空的热力图
        status = "未检测到带钢"  # 初始化默认状态
        algorithm_time = 0
        dehazed_frame = frame.copy()  # 初始化去雾后的帧
        contour_frame = frame.copy()  # 初始化轮廓提取后的帧
        wave_height = 0.0  # 浪高
        wave_width = 0.0  # 浪宽
        wave_level = "无"  # 浪形等级

        # --- 第一步：图像预处理与自适应二值化 ---
        # 1. 暗通道去雾
        dehazed_frame = self.dehaze(frame)
        # 2. CLAHE 对比度增强
        dehazed_frame = self.enhance_contrast(dehazed_frame)

        # 3. 使用 HSV 色彩空间，提取高温带钢的亮色区域（红/橙/黄/白）
        hsv = cv2.cvtColor(dehazed_frame, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]  # 色调 (0-180)
        s_channel = hsv[:, :, 1]  # 饱和度 (0-255)
        v_channel = hsv[:, :, 2]  # 亮度 (0-255)

        # 4. 双边滤波：平滑噪声的同时保留边缘
        v_channel_filtered = cv2.bilateralFilter(v_channel, 9, 75, 75)

        # 5. 针对高温带钢的多条件颜色筛选策略：
        #    高温带钢特征：非常高亮度 + 明显的红色/橙色色调
                
        # 【关键】大幅提高亮度阈值，只保留真正高温的亮白色/红色区域
        # 条件A：非常高亮度区域（V > 180）
        bright_mask = v_channel_filtered > 180
                
        # 条件B1：低饱和度高亮度（纯白色系，V必须很高）
        white_like = (s_channel < 80) & (v_channel_filtered > 180)
                
        # 条件B2：红色调 (H: 0-10 或 170-180) + 中高饱和度 + 非常高亮度
        red_tone = ((h_channel <= 10) | (h_channel >= 170)) & (s_channel > 60) & (v_channel_filtered > 160)
                
        # 条件B3：橙色调 (H: 11-25) + 中高饱和度 + 非常高亮度
        orange_tone = (h_channel > 10) & (h_channel <= 25) & (s_channel > 50) & (v_channel_filtered > 160)
                
        # 条件B4：黄色调 (H: 26-35) + 中等饱和度 + 高亮度（收紧条件）
        yellow_tone = (h_channel > 25) & (h_channel <= 35) & (s_channel > 40) & (v_channel_filtered > 170)
                
        # 组合所有条件：只要满足任一颜色条件即可
        color_mask = white_like | red_tone | orange_tone | yellow_tone
                
        # 最终掩码：必须是高亮度区域且符合颜色特征
        mask = (bright_mask & color_mask).astype(np.uint8) * 255

        # 【重要】不再使用纯亮度阈值的回退策略，避免误检背景
        # 如果颜色筛选检测到的区域太小，说明当前帧确实没有带钢或带钢颜色不明显
        # 此时应该保持较小的检测区域，而不是放宽到所有亮色区域
        if np.sum(mask) < 5000:  # 如果检测到的区域太小
            # 不执行回退，保持当前的颜色筛选结果
            # 这样可以避免将背景的传送带反光等误判为带钢
            pass

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
        depth_map = np.zeros((h, w), dtype=np.uint8)  # 初始化空的高度图矩阵
        largest_contour = None  # 初始化最大轮廓

        if contours:
            # 找到画面中面积最大的连通域
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            x, y, bw, bh = cv2.boundingRect(largest_contour)

            # 计算轮廓实心度（面积与凸包面积的比例）
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # 【增强】条件确认：必须同时满足多个条件才确认为真正的带钢
            # 1. 面积够大 (>50000) - 根据调试数据，背景干扰约27000-29000，带钢约108000
            # 2. 宽度占画面的 20% 以上
            # 3. 实心度大于 0.7
                        
            # 【调试信息】打印检测结果，帮助诊断
            print(f"[DEBUG] Contour check - Area: {area:.0f}, Width ratio: {bw / w:.2f}, Solidity: {solidity:.2f}")
                        
            # 通过面积阈值有效区分带钢和背景干扰
            if area > 50000 and bw > w * 0.2 and solidity > 0.7:

                # --- 第四步：触发 ROI 锁定系统 ---
                if not self.roi_locked:
                    margin = 30  # 上下预留 30 像素的容错空间
                    self.roi_top = max(0, y - margin)
                    self.roi_bottom = min(h, y + bh + margin)
                    self.roi_locked = True
                    self.stable_frames = 0  # 新带钢咬钢，重置稳定计数
                    # 清空上一根带钢遗留的历史振幅数据
                    self.ws_amp_history.clear()
                    self.ds_amp_history.clear()

                self.stable_frames += 1  # 稳定帧数递增

                # --- 第五步：提取带钢精确边缘曲线 ---
                plate_mask = mask[y:y + bh, x:x + bw]
                valid_cols = np.any(plate_mask > 0, axis=0)  # 找出有非黑像素的有效列

                if np.sum(valid_cols) > 50:  # 有效列足够多才进行计算
                    # argmax 寻找每列第一个非零像素点，获取最原始的上下边缘坐标
                    top_edge_raw = np.argmax(plate_mask[:, valid_cols], axis=0)
                    bottom_edge_raw = bh - np.argmax(plate_mask[::-1, valid_cols], axis=0)

                    # 【增强】斜率约束验证：过滤掉传送带圆柱体等干扰
                    # 带钢轮廓应该相对平滑，相邻点的斜率变化不会太大
                    def validate_edge_by_slope(edge_curve, max_slope_change=15):
                        """
                        通过斜率变化验证边缘的有效性
                        如果相邻点的斜率变化超过阈值，说明可能是传送带反光等干扰
                        返回：(是否有效, 清理后的边缘曲线)
                        """
                        if len(edge_curve) < 3:
                            return False, edge_curve

                        # 计算一阶差分（近似斜率）
                        diffs = np.diff(edge_curve)

                        # 计算斜率的变化量（二阶差分）
                        slope_changes = np.abs(np.diff(diffs))

                        # 找出斜率突变的位置
                        abrupt_changes = np.where(slope_changes > max_slope_change)[0]

                        # 如果超过20%的点有突变，判定为无效边缘（适度宽松）
                        if len(abrupt_changes) > len(edge_curve) * 0.20:
                            return False, edge_curve

                        # 对斜率突变的点进行修复
                        cleaned_edge = edge_curve.copy()
                        for idx in abrupt_changes:
                            # 用前后各5个点的中位数替换突变点
                            start = max(0, idx - 5)
                            end = min(len(edge_curve), idx + 6)
                            cleaned_edge[idx] = np.median(edge_curve[start:end])

                        return True, cleaned_edge

                    # 验证上下边缘的有效性（使用适度宽松的参数）
                    top_valid, top_edge_cleaned = validate_edge_by_slope(top_edge_raw, max_slope_change=15)
                    bottom_valid, bottom_edge_cleaned = validate_edge_by_slope(bottom_edge_raw, max_slope_change=15)

                    # 【调试信息】打印边缘验证结果
                    print(
                        f"[DEBUG] Edge validation - Top: {'OK' if top_valid else 'FAIL'}, Bottom: {'OK' if bottom_valid else 'FAIL'}")

                    # 【关键】如果边缘验证失败，说明这个轮廓可能不是真正的带钢
                    # 此时应该拒绝绘制轮廓，避免误检
                    if not top_valid or not bottom_valid:
                        # 边缘不规则，可能是背景干扰，不绘制轮廓
                        largest_contour = None  # 清除轮廓，阻止后续绘制
                        status = "未检测到带钢"
                    else:
                        # 边缘验证通过，继续处理
                        # 调用工具函数，对边缘进行一维滤波磨皮
                        top_edge_smooth = self.smooth_curve(top_edge_cleaned, window_size=21)
                        bottom_edge_smooth = self.smooth_curve(bottom_edge_cleaned, window_size=21)

                        # --- 第六步：过渡区延时启动与状态判定 ---
                        # 设定 30 帧 (约 1 秒) 作为缓冲期。等待带钢彻底进入画面且水雾散开
                        is_stable_body = (self.stable_frames > 30)

                        if is_stable_body:
                            # 1. 计算当前单帧的真实振幅（已去除畸变）
                            raw_ws_amp, ws_wave_height, ws_wave_width = self.calculate_true_amplitude(top_edge_smooth)
                            raw_ds_amp, ds_wave_height, ds_wave_width = self.calculate_true_amplitude(
                                bottom_edge_smooth)

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
                            t = max(0, min(bh - 1, t))  # 越界保护
                            b = max(0, min(bh - 1, b))
                            if b > t:
                                clean_mask[t:b, col] = 255  # 填充白色实心

                        # 构造一个立体的纵向高度渐变（让图像有 3D 圆柱表面的隆起感）
                        gradient = np.linspace(50, 200, bh, dtype=np.float32)
                        gradient_2d = np.tile(gradient, (bw, 1)).T
                        simulated_depth = np.clip(gradient_2d, 0, 255).astype(np.uint8)

                        # 将深度贴图精确地覆盖到我们刚刚画好的 Clean Mask 上
                        depth_map[y:y + bh, x:x + bw] = cv2.bitwise_and(simulated_depth, simulated_depth,
                                                                        mask=clean_mask)

                else:
                    # 有效列不足，无法提取可靠的边缘，拒绝绘制
                    largest_contour = None
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

        # 生成轮廓提取后的帧 - 只在确认的带钢上绘制标记
        contour_frame = dehazed_frame.copy()

        # 【关键保护】只有当 largest_contour 不为 None 且状态不是"未检测到带钢"时才绘制
        # 这样可以确保所有拒绝的情况都不会绘制轮廓
        if largest_contour is not None and status != "未检测到带钢":
            # 只绘制最大的那个轮廓（真正的带钢），用绿色线条
            cv2.drawContours(contour_frame, [largest_contour], -1, (0, 255, 0), 3)
            # 绘制边界框，用红色
            x, y, bw, bh = cv2.boundingRect(largest_contour)
            cv2.rectangle(contour_frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)

            # 【新增】如果已经提取了边缘，绘制平滑后的边缘曲线
            if 'top_edge_smooth' in locals() and 'bottom_edge_smooth' in locals() and 'valid_cols' in locals():
                valid_indices = np.where(valid_cols)[0]
                if len(valid_indices) > 0:
                    # 绘制上边缘（蓝色）
                    for i in range(len(valid_indices) - 1):
                        pt1 = (x + valid_indices[i], y + int(top_edge_smooth[i]))
                        pt2 = (x + valid_indices[i + 1], y + int(top_edge_smooth[i + 1]))
                        cv2.line(contour_frame, pt1, pt2, (255, 0, 0), 2)

                    # 绘制下边缘（青色）
                    for i in range(len(valid_indices) - 1):
                        pt1 = (x + valid_indices[i], y + int(bottom_edge_smooth[i]))
                        pt2 = (x + valid_indices[i + 1], y + int(bottom_edge_smooth[i + 1]))
                        cv2.line(contour_frame, pt1, pt2, (255, 255, 0), 2)

            # 添加文字标注
            cv2.putText(contour_frame, f'Status: {status}', (x, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if wave_height > 0:
                cv2.putText(contour_frame, f'Wave: {wave_height:.1f}mm ({wave_level})',
                            (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 【新增】显示边缘验证状态
            if 'top_valid' in locals() and 'bottom_valid' in locals():
                edge_status = f"Edge: T={'OK' if top_valid else 'WARN'} B={'OK' if bottom_valid else 'WARN'}"
                cv2.putText(contour_frame, edge_status, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
        heatmap, status, algorithm_time, dehazed_frame, contour_frame, wave_height, wave_width, wave_level, edge_data = self.process_frame(
            frame)

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