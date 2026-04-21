import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QListWidget, 
                             QGridLayout, QFileDialog, QSizePolicy, QSlider, 
                             QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt, QTime, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from src.wave_detector import WaveDetector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 0. 自定义可点击的 Label
# ==========================================
class ClickableLabel(QLabel):
    # 定义一个点击信号
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        # 当鼠标左键点击时发射信号
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

# ==========================================
# 1. 视频与算法处理线程 (后台执行)
# ==========================================
class VideoThread(QThread):
    update_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool, int, int, float, float, float, str, dict)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_running = True
        self.current_frame = 0
        self.total_frames = 0
        self.low_resolution = False
        
        # 【新增】实例化算法类
        self.detector = WaveDetector()

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        cap = self.cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 记录上一帧的状态，防止日志被同一报警疯狂刷屏
        last_status = "平直" 
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                continue
            
            # 如果是低分辨率模式，缩小帧尺寸以提高处理速度
            original_frame = frame.copy()
            if self.low_resolution:
                frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            
            # 传进去原图，拿回来热力图、文字状态、算法处理时间、去雾后的帧和轮廓提取后的帧
            heatmap, status, algorithm_time, dehazed_frame, contour_frame, wave_height, wave_width, wave_level, edge_data = self.detector.process_frame(frame)
            
            # 如果是低分辨率模式，恢复帧尺寸
            if self.low_resolution:
                heatmap = cv2.resize(heatmap, (original_frame.shape[1], original_frame.shape[0]))
                dehazed_frame = cv2.resize(dehazed_frame, (original_frame.shape[1], original_frame.shape[0]))
                contour_frame = cv2.resize(contour_frame, (original_frame.shape[1], original_frame.shape[0]))
                frame = original_frame
            
            # 判断是否需要打印日志 (只有状态从平直突变成报警，或者报警类型变化时才记录)
            log_flag = False
            if "报警" in status and status != last_status:
                log_flag = True
            
            last_status = status # 更新状态记忆
            self.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 发送给 UI 界面渲染
            self.update_signal.emit(frame, dehazed_frame, contour_frame, heatmap, status, log_flag, self.current_frame, self.total_frames, algorithm_time, wave_height, wave_width, wave_level, edge_data)
            
            self.msleep(30)

        if hasattr(self, 'cap'):
            self.cap.release()
        
    def stop(self):
        self.is_running = False
        self.wait()
    
    def update_parameters(self, omega, t0, binary_threshold):
        """更新detector的参数"""
        if hasattr(self, 'detector'):
            self.detector.set_parameters(omega, t0, binary_threshold)
    
    def set_frame(self, frame_index):
        """设置视频帧的位置"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self.current_frame = frame_index
    
    def set_low_resolution(self, low_res):
        """设置是否使用低分辨率处理"""
        self.low_resolution = low_res

# ==========================================
# 2. 主 UI 界面
# ==========================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None # 初始化时不启动线程
        self.video_path = ""
        self.log_history = [] # 日志历史记录
        self.initUI()

    def initUI(self):
        self.setWindowTitle('带钢动态轮廓与浪形检测系统')
        self.resize(1400, 900)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")

        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 顶部布局：视频和日志
        top_layout = QHBoxLayout()
        
        # 视频区域布局
        video_layout = QGridLayout()
        
        # 原始监控视频
        self.video_label = ClickableLabel("点击此处\n导入现场监控视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px dashed #555; font-size: 20px; color: #888;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 设置固定大小（增大视频区域）
        self.video_label.setMinimumSize(500, 300)
        self.video_label.setMaximumSize(600, 350)
        
        # 去雾后的视频
        self.dehazed_label = QLabel("等待视频输入...")
        self.dehazed_label.setAlignment(Qt.AlignCenter)
        self.dehazed_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
        self.dehazed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 设置固定大小（增大视频区域）
        self.dehazed_label.setMinimumSize(500, 300)
        self.dehazed_label.setMaximumSize(600, 350)
        
        # 轮廓提取与目标确认
        self.contour_label = QLabel("等待视频输入...")
        self.contour_label.setAlignment(Qt.AlignCenter)
        self.contour_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
        self.contour_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 设置固定大小（增大视频区域）
        self.contour_label.setMinimumSize(500, 300)
        self.contour_label.setMaximumSize(600, 350)
        
        # 轮廓检测热力图
        self.heatmap_label = QLabel("等待视频输入...")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
        self.heatmap_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 设置固定大小（增大视频区域）
        self.heatmap_label.setMinimumSize(500, 300)
        self.heatmap_label.setMaximumSize(600, 350)
        
        # 视频控制区域
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.frame_label = QLabel("0 / 0")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setValue(0)
        
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.frame_label)
        control_layout.addWidget(self.frame_slider)
        control_layout.addStretch()
        
        # 添加视频区域到布局（2x2布局）
        video_layout.addWidget(QLabel("原始监控视频 (点击画面切换视频)"), 0, 0)
        video_layout.addWidget(QLabel("去雾后的视频"), 0, 1)
        video_layout.addWidget(QLabel("轮廓提取与目标确认"), 2, 0)
        video_layout.addWidget(QLabel("轮廓检测热力图"), 2, 1)
        video_layout.addWidget(self.video_label, 1, 0)
        video_layout.addWidget(self.dehazed_label, 1, 1)
        video_layout.addWidget(self.contour_label, 3, 0)
        video_layout.addWidget(self.heatmap_label, 3, 1)
        # 将视频控制区域放在所有视频下方
        video_layout.addLayout(control_layout, 4, 0, 1, 2)
        
        # 设置布局比例
        video_layout.setRowStretch(1, 1)
        video_layout.setRowStretch(3, 1)
        video_layout.setRowStretch(4, 0)  # 视频控制区域不占太多空间
        video_layout.setColumnStretch(0, 1)
        video_layout.setColumnStretch(1, 1)
        
        # 右侧面板：日志和参数设置
        right_layout = QVBoxLayout()
        
        # 日志区域（缩小）
        log_group = QGroupBox("系统检测日志")
        log_group.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("background-color: #2D2D2D; font-size: 12px; border: 1px solid #555;")
        # 设置固定大小（缩小日志区域）
        self.log_list.setMinimumSize(250, 300)
        self.log_list.setMaximumSize(300, 400)
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_list)
        
        # 回溯按钮
        backtrack_layout = QHBoxLayout()
        self.backtrack_button = QPushButton("日志回溯")
        backtrack_layout.addWidget(self.backtrack_button)
        backtrack_layout.addStretch()
        log_layout.addLayout(backtrack_layout)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # 参数设置面板
        param_group = QGroupBox("图像处理参数")
        param_group.setStyleSheet("font-size: 14px; font-weight: bold;")
        param_layout = QGridLayout()
        
        # 去雾参数 - Omega
        self.omega_label = QLabel("去雾强度:")
        self.omega_slider = QSlider(Qt.Horizontal)
        self.omega_slider.setMinimum(50)
        self.omega_slider.setMaximum(95)
        self.omega_slider.setValue(80)
        self.omega_value = QLabel("0.80")
        self.omega_desc = QLabel("(值越小效果越强)")
        self.omega_desc.setStyleSheet("font-size: 10px; color: #888;")
        
        # 去雾参数 - t0
        self.t0_label = QLabel("透射率:")
        self.t0_slider = QSlider(Qt.Horizontal)
        self.t0_slider.setMinimum(10)
        self.t0_slider.setMaximum(30)
        self.t0_slider.setValue(25)
        self.t0_value = QLabel("0.25")
        self.t0_desc = QLabel("(值越大效果越弱)")
        self.t0_desc.setStyleSheet("font-size: 10px; color: #888;")
        
        # 二值化阈值
        self.threshold_label = QLabel("轮廓阈值:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(100)
        self.threshold_slider.setMaximum(150)
        self.threshold_slider.setValue(120)
        self.threshold_value = QLabel("120")
        self.threshold_desc = QLabel("(值越大轮廓越亮)")
        self.threshold_desc.setStyleSheet("font-size: 10px; color: #888;")
        
        param_layout.addWidget(self.omega_label, 0, 0)
        param_layout.addWidget(self.omega_slider, 0, 1)
        param_layout.addWidget(self.omega_value, 0, 2)
        param_layout.addWidget(self.omega_desc, 0, 3)
        
        param_layout.addWidget(self.t0_label, 1, 0)
        param_layout.addWidget(self.t0_slider, 1, 1)
        param_layout.addWidget(self.t0_value, 1, 2)
        param_layout.addWidget(self.t0_desc, 1, 3)
        
        param_layout.addWidget(self.threshold_label, 2, 0)
        param_layout.addWidget(self.threshold_slider, 2, 1)
        param_layout.addWidget(self.threshold_value, 2, 2)
        param_layout.addWidget(self.threshold_desc, 2, 3)
        
        param_group.setLayout(param_layout)
        right_layout.addWidget(param_group)
        
        # 将视频布局和右侧布局添加到顶部布局
        top_layout.addLayout(video_layout, 3)
        top_layout.addLayout(right_layout, 1)
        
        # 底部检测结果区域
        result_layout = QGridLayout()
        
        # 左侧：状态和时间信息
        left_result_layout = QVBoxLayout()
        self.status_label = QLabel("当前状态：请先导入视频")
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #888;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        time_layout = QHBoxLayout()
        self.algorithm_time_label = QLabel("算法处理时间：0 ms")
        self.algorithm_time_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #888;")
        time_layout.addWidget(self.algorithm_time_label)
        time_layout.addStretch()
        
        left_result_layout.addWidget(self.status_label)
        left_result_layout.addLayout(time_layout)
        
        # 右侧：浪形信息
        right_result_layout = QVBoxLayout()
        self.wave_type_label = QLabel("浪形类型：无")
        self.wave_type_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #888;")
        
        self.wave_size_label = QLabel("浪形大小：无")
        self.wave_size_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #888;")
        
        right_result_layout.addWidget(self.wave_type_label)
        right_result_layout.addWidget(self.wave_size_label)
        right_result_layout.addStretch()
        
        # 底部：边缘变化展示窗口
        edge_layout = QVBoxLayout()
        edge_group = QGroupBox("卷浪类型上下边缘变化")
        edge_group.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        # 创建matplotlib画布
        self.fig = Figure(figsize=(10, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #2D2D2D;")
        
        edge_layout.addWidget(self.canvas)
        edge_group.setLayout(edge_layout)
        
        # 添加到主结果布局
        result_layout.addLayout(left_result_layout, 0, 0, 1, 2)
        result_layout.addLayout(right_result_layout, 0, 2, 1, 2)
        result_layout.addWidget(edge_group, 1, 0, 1, 4)
        
        # 添加所有布局到主布局
        main_layout.addLayout(top_layout)
        main_layout.addLayout(result_layout)
        
        # 设置布局比例
        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 1)
        
        self.setLayout(main_layout)

        # 绑定点击事件到导入视频的方法
        self.video_label.clicked.connect(self.open_video_dialog)
        self.frame_slider.sliderPressed.connect(self.slider_pressed)
        self.frame_slider.sliderMoved.connect(self.slider_moved)
        self.frame_slider.sliderReleased.connect(self.slider_released)
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.backtrack_button.clicked.connect(self.backtrack_log)
        
        # 参数设置滑块信号
        self.omega_slider.valueChanged.connect(self.update_omega)
        self.t0_slider.valueChanged.connect(self.update_t0)
        self.threshold_slider.valueChanged.connect(self.update_binary_threshold)

    # 打开文件对话框导入视频
    def open_video_dialog(self):
        # 只允许选择常见视频格式
        video_path, _ = QFileDialog.getOpenFileName(
            parent=self, 
            caption="选择监控视频", 
            directory="", 
            filter="视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if video_path:
            # 清除提示文字和虚线边框
            self.video_label.setText("")
            self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
            self.status_label.setText("当前状态：初始化算法...")
            
            # 如果之前有视频在播放，先安全停止
            if self.thread is not None and self.thread.isRunning():
                self.thread.stop()

            # 启动新线程
            self.thread = VideoThread(video_path)
            self.thread.update_signal.connect(self.update_frame)
            self.thread.start()

    # 更新界面
    def update_frame(self, frame, dehazed_frame, contour_frame, heatmap, status, log_flag, current_frame, total_frames, algorithm_time, wave_height, wave_width, wave_level, edge_data):
        self.display_image(frame, self.video_label)
        self.display_image(dehazed_frame, self.dehazed_label)
        self.display_image(contour_frame, self.contour_label)
        self.display_image(heatmap, self.heatmap_label)
        
        self.status_label.setText(f"当前状态：{status}")
        if "报警" in status:
            self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF3333; background-color: #440000;")
        else:
            self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FF00; background-color: transparent;")
        
        # 更新浪形信息
        if "未检测到带钢" in status:
            self.wave_type_label.setText("浪形类型：无")
            self.wave_size_label.setText("浪形大小：无")
        elif "平直" in status:
            self.wave_type_label.setText("浪形类型：平直")
            self.wave_size_label.setText("浪形大小：无")
        elif "过渡区" in status:
            self.wave_type_label.setText("浪形类型：过渡区")
            self.wave_size_label.setText("浪形大小：无")
        elif "双边浪" in status:
            self.wave_type_label.setText("浪形类型：双边浪")
            self.wave_size_label.setText(f"浪形大小：高 (高: {wave_height:.2f} mm, 宽: {wave_width:.2f} mm)")
        elif "WS侧单边浪" in status:
            self.wave_type_label.setText("浪形类型：WS侧单边浪")
            self.wave_size_label.setText(f"浪形大小：中 (高: {wave_height:.2f} mm, 宽: {wave_width:.2f} mm)")
        elif "DS侧单边浪" in status:
            self.wave_type_label.setText("浪形类型：DS侧单边浪")
            self.wave_size_label.setText(f"浪形大小：中 (高: {wave_height:.2f} mm, 宽: {wave_width:.2f} mm)")

        # 更新视频条
        self.frame_label.setText(f"{current_frame} / {total_frames}")
        self.frame_slider.setMaximum(total_frames)
        self.frame_slider.setValue(current_frame)
        
        # 更新算法处理时间
        self.algorithm_time_label.setText(f"算法处理时间：{algorithm_time:.2f} ms")
        if algorithm_time > 200:
            self.algorithm_time_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF3333;")
        elif algorithm_time > 100:
            self.algorithm_time_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFAA00;")
        else:
            self.algorithm_time_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00FF00;")

        if log_flag:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            # 计算视频时间（假设30fps）
            fps = 30
            video_time = current_frame / fps
            minutes = int(video_time // 60)
            seconds = int(video_time % 60)
            milliseconds = int((video_time % 1) * 1000)
            log_msg = f"[{current_time}] 检测到异常：带钢发生 {status} (Frame: {current_frame}, Time: {minutes:02d}:{seconds:02d}.{milliseconds:03d})"
            self.log_list.insertItem(0, log_msg)
            self.log_history.append((current_time, status, current_frame))
            if self.log_list.count() > 100:
                self.log_list.takeItem(self.log_list.count() - 1)
        
        # 更新边缘变化曲线
        self.update_edge_plot(edge_data)

    # 渲染图像
    def display_image(self, img_array, label_widget):
        rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # 获取当前 Label 的真实大小，进行等比例缩放
        scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)

    # 视频条拖动开始事件
    def slider_pressed(self):
        # 拖动开始，设置低分辨率处理
        if self.thread is not None:
            self.thread.set_low_resolution(True)
    
    # 视频条拖动过程事件
    def slider_moved(self, value):
        # 更新帧标签
        if self.thread is not None:
            total_frames = self.thread.total_frames
            self.frame_label.setText(f"{value} / {total_frames}")
            # 直接设置帧位置，不停止线程
            self.thread.set_frame(value)
    
    # 视频条拖动结束事件
    def slider_released(self):
        if self.thread is not None:
            # 恢复正常分辨率
            self.thread.set_low_resolution(False)

    # 播放视频
    def play_video(self):
        if self.thread is not None and not self.thread.isRunning():
            self.thread.start()

    # 暂停视频
    def pause_video(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()

    # 停止视频
    def stop_video(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        # 重置视频到开始位置
        if self.thread is not None:
            self.thread.set_frame(0)

    # 日志回溯
    def backtrack_log(self):
        if self.log_history:
            # 简单实现：跳转到最后一条日志对应的帧
            last_log = self.log_history[-1]
            current_time, status, frame_index = last_log
            if self.thread is not None:
                self.thread.set_frame(frame_index)
                # 显示回溯信息
                self.status_label.setText(f"回溯到：{current_time} - {status}")
                self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFF00; background-color: #444400;")
    
    # 更新浪形振幅阈值
    def update_threshold(self, value):
        threshold = value / 10.0
        self.threshold_value.setText(f"{threshold:.1f}")
        if self.thread is not None and hasattr(self.thread.detector, 'wave_amplitude_threshold'):
            self.thread.detector.wave_amplitude_threshold = threshold
    
    # 更新滑动窗口大小
    def update_window(self, value):
        self.window_value.setText(f"{value}")
        if self.thread is not None and hasattr(self.thread.detector, 'history_frames'):
            self.thread.detector.history_frames = value
            # 重新初始化历史队列
            self.thread.detector.ws_amp_history = deque(maxlen=value)
            self.thread.detector.ds_amp_history = deque(maxlen=value)
    
    # 更新像素到毫米转换系数
    def update_omega(self, value):
        omega = value / 100.0
        self.omega_value.setText(f"{omega:.2f}")
        # 更新detector的omega参数
        if self.thread is not None and self.thread.isRunning():
            t0 = self.t0_slider.value() / 100.0
            binary_threshold = self.threshold_slider.value()
            self.thread.update_parameters(omega, t0, binary_threshold)
    
    def update_t0(self, value):
        t0 = value / 100.0
        self.t0_value.setText(f"{t0:.2f}")
        # 更新detector的t0参数
        if self.thread is not None and self.thread.isRunning():
            omega = self.omega_slider.value() / 100.0
            binary_threshold = self.threshold_slider.value()
            self.thread.update_parameters(omega, t0, binary_threshold)
    
    def update_binary_threshold(self, value):
        self.threshold_value.setText(f"{value}")
        # 更新detector的二值化阈值参数
        if self.thread is not None and self.thread.isRunning():
            omega = self.omega_slider.value() / 100.0
            t0 = self.t0_slider.value() / 100.0
            self.thread.update_parameters(omega, t0, value)
    
    def update_edge_plot(self, edge_data):
        """更新边缘变化曲线"""
        # 清除之前的图像
        self.fig.clear()
        
        # 设置背景颜色
        self.fig.patch.set_facecolor('#2D2D2D')
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#2D2D2D')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        
        # 检查边缘数据是否存在
        if edge_data and ('top_edge_smooth' in edge_data) and ('bottom_edge_smooth' in edge_data):
            top_edge = edge_data['top_edge_smooth']
            bottom_edge = edge_data['bottom_edge_smooth']
            
            if top_edge and bottom_edge:
                # 绘制上下边缘曲线
                ax.plot(top_edge, label='Top Edge', color='green')
                ax.plot(bottom_edge, label='Bottom Edge', color='red')
                ax.set_xlabel('Pixel Column', color='white')
                ax.set_ylabel('Pixel Row', color='white')
                ax.legend(facecolor='#2D2D2D', edgecolor='white', labelcolor='white')
                ax.set_title('Wave Edge Variation', color='white')
        else:
            ax.set_xlabel('Pixel Column', color='white')
            ax.set_ylabel('Pixel Row', color='white')
            ax.set_title('Wave Edge Variation', color='white')
            ax.text(0.5, 0.5, 'No Edge Data', ha='center', va='center', color='white', transform=ax.transAxes)
        
        # 更新画布
        self.canvas.draw()

    # 关闭窗口时清理线程
    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())