import sys
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QListWidget, 
                             QGridLayout, QFileDialog, QSizePolicy, QSlider, 
                             QPushButton, QHBoxLayout, QVBoxLayout, QListWidgetItem, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from src.wave_detector import WaveDetector

# ==========================================
# 【基础交互组件】可点击的标签，用于触发视频导入
# ==========================================
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

# ==========================================
# 【核心模块四】异步视频流时序控制 (Asynchronous Timeline Control)
# 采用 QThread 将高负载机器视觉运算与主 UI 线程深度解耦
# ==========================================
class VideoThread(QThread):
    # 跨线程通信信号：分别用于更新四联屏画面、初始化 UI 引擎、跨线程日志状态同步
    update_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool, int)
    init_signal = pyqtSignal(int, int, int)
    clear_log_signal = pyqtSignal() # 跨线程日志自我净化信号 (Auto-Purge on Hot-Swap)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_running = True
        self.is_paused = False
        self.seek_frame = -1 
        self.force_refresh = False 
        
        self.detector = WaveDetector()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames <= 0: total_frames = 1000 
        
        # 将视频物理尺寸抛给主线程，用于触发“自适应视窗缩放引擎”
        self.init_signal.emit(total_frames, width, height)
        
        last_status = "未检测到带钢" 
        last_perspective = "未知"
        
        while cap.isOpened() and self.is_running:
            need_read = False
            
            # 时序精确控制：支持非线性随机拖拽寻址
            if self.seek_frame != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                self.seek_frame = -1
                need_read = True 
            # 暂停状态热重载机制 (Hot-Reloading Mechanism)
            elif not self.is_paused or self.force_refresh:
                if self.is_paused and self.force_refresh:
                    # 倒退一帧重新读取，实现参数修改的零延迟视觉反馈
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                need_read = True

            if need_read:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 执行底层机器视觉算法流水线
                heatmap, status, annotated_frame, defogged_frame = self.detector.process_frame(frame)

                # 跨线程日志自我净化拦截器：监测到视角发生热纠错时，清空误判日志
                current_perspective = self.detector.current_perspective
                if last_perspective != "未知" and current_perspective != "未知" and last_perspective != current_perspective:
                    self.clear_log_signal.emit()
                last_perspective = current_perspective

                log_flag = False
                if "报警" in status and status != last_status:
                    log_flag = True
                last_status = status

                self.update_signal.emit(frame, defogged_frame, annotated_frame, heatmap, status, log_flag, current_frame)
                self.force_refresh = False 
            
            self.msleep(30) 

        cap.release()
        
    def set_frame(self, frame_idx):
        self.seek_frame = frame_idx

    def refresh_current_frame(self):
        self.force_refresh = True

    def stop(self):
        self.is_running = False
        self.wait()

# ==========================================
# 【核心模块一】工业级交互界面主窗体
# ==========================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None 
        self.is_slider_dragging = False
        self.was_paused_before_drag = False
        self.total_frames = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('带钢动态轮廓提取与浪形检测系统 (工业调参版)')
        self.resize(1500, 950)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")

        # ------------------------------------------
        # 1. 动态自适应“田字格”四联屏视窗构建
        # ------------------------------------------
        # 视图 1：原始监控视频 (Ground Truth)
        self.video_label = ClickableLabel("点击此处\n导入现场监控视频")
        self.setup_label_style(self.video_label, dashed=True)
        self.video_label.clicked.connect(self.open_video_dialog)
        
        # 视图 2：背景物理隔离与去雾效果 (Valid Mask)
        self.defogged_label = QLabel("等待视频接入...")
        self.setup_label_style(self.defogged_label)

        # 视图 3：带钢轮廓提取与浪形检测区域标定 (Contour & ROI)
        self.annotated_label = QLabel("等待视频接入...")
        self.setup_label_style(self.annotated_label)

        # 视图 4：带钢形貌伪色彩渲染 (3D Pseudo-color Rendering)
        self.heatmap_label = QLabel("等待视频接入...")
        self.setup_label_style(self.heatmap_label)

        # ------------------------------------------
        # 2. 结构化报警日志面板构建 (Structured Log System)
        # ------------------------------------------
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("""
            QListWidget {
                background-color: #2D2D2D; 
                font-size: 14px; 
                border: 1px solid #555; 
                padding: 5px;
            }
            QListWidget::item {
                padding-top: 5px;
                padding-bottom: 5px;
            }
        """)
        # 绑定历史帧快速定位功能 (Frame Index Tracing)
        self.log_list.itemDoubleClicked.connect(self.jump_to_log_frame)

        self.status_label = QLabel("当前状态：请先导入视频")
        self.status_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #888;")
        self.status_label.setAlignment(Qt.AlignCenter)

        # ------------------------------------------
        # 3. 时序精确控制组件构建
        # ------------------------------------------
        self.play_button = QPushButton("▶ 播放/暂停")
        self.play_button.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        self.play_button.setFixedSize(120, 35)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False) 

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet("QSlider::handle:horizontal {background: #00FF00; width: 12px; margin: -4px 0; border-radius: 6px;} QSlider::groove:horizontal {height: 4px; background: #555;}")
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.sliderReleased.connect(self.slider_released)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00FF00; padding: 0 10px;")

        video_control_layout = QHBoxLayout()
        video_control_layout.addWidget(self.play_button)
        video_control_layout.addWidget(self.slider)
        video_control_layout.addWidget(self.frame_label)

        # ------------------------------------------
        # 4. 算法参数实时整定控制台 (Real-time Parameter Tuning Console)
        # ------------------------------------------
        param_group = QGroupBox("算法参数实时整定控制台")
        param_group.setStyleSheet("QGroupBox { border: 1px solid #666; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        param_layout = QHBoxLayout()

        # 四大核心工业控制滑块，下发参数即时生效
        self.sl_defog, self.lbl_defog = self.create_param_slider("去水雾强度", 0, 30, 5)
        self.sl_gray, self.lbl_gray = self.create_param_slider("轮廓提取灰度阈值", 100, 255, 200)
        self.sl_trim, self.lbl_trim = self.create_param_slider("浪形检测切除比例(%)", 0, 30, 15, scale=1.0)
        self.sl_amp, self.lbl_amp = self.create_param_slider("浪形检测报警阈值", 10, 150, 50, scale=10.0)

        param_layout.addLayout(self.sl_defog)
        param_layout.addLayout(self.sl_gray)
        param_layout.addLayout(self.sl_trim)
        param_layout.addLayout(self.sl_amp)
        param_group.setLayout(param_layout)

        # ------------------------------------------
        # 5. 全局网格拉伸映射与布局组装
        # ------------------------------------------
        grid = QGridLayout()
        
        grid.addWidget(QLabel("原始监控视频"), 0, 0)
        grid.addWidget(QLabel("背景物理隔离与去雾效果"), 0, 1)
        grid.addWidget(QLabel(">>> 报警抓拍日志 (双击跳转) <<<"), 0, 2)
        
        grid.addWidget(self.video_label, 1, 0)
        grid.addWidget(self.defogged_label, 1, 1)
        grid.addWidget(self.log_list, 1, 2, 3, 1) 
        
        grid.addWidget(QLabel("带钢轮廓提取与浪形检测区域标定"), 2, 0)
        grid.addWidget(QLabel("带钢形貌伪色彩渲染"), 2, 1)
        
        grid.addWidget(self.annotated_label, 3, 0)
        grid.addWidget(self.heatmap_label, 3, 1)

        grid.addWidget(self.status_label, 4, 0, 1, 3)
        grid.addLayout(video_control_layout, 5, 0, 1, 3) 
        grid.addWidget(param_group, 6, 0, 1, 3)

        # 精密分配网格拉伸系数，压榨空白区域以配合无黑边自适应引擎
        grid.setColumnStretch(0, 5) 
        grid.setColumnStretch(1, 5) 
        grid.setColumnStretch(2, 3) 
        
        grid.setRowStretch(0, 1) 
        grid.setRowStretch(1, 10) 
        grid.setRowStretch(2, 1)
        grid.setRowStretch(3, 10) 
        grid.setRowStretch(4, 1)
        grid.setRowStretch(5, 1)
        grid.setRowStretch(6, 1)

        self.setLayout(grid)

    def setup_label_style(self, label, dashed=False):
        label.setAlignment(Qt.AlignCenter)
        border = "2px dashed #555;" if dashed else "1px solid #333;"
        label.setStyleSheet(f"background-color: #000000; border: {border}; color: #888; font-size: 16px;")
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def create_param_slider(self, name, min_val, max_val, default_val, scale=1.0):
        layout = QVBoxLayout()
        label = QLabel(f"{name}: {default_val/scale if scale != 1.0 else default_val}")
        label.setAlignment(Qt.AlignCenter)
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        
        # 绑定参数整定事件回调
        slider.valueChanged.connect(lambda val: self.on_param_changed(slider, label, name, scale))
        
        layout.addWidget(label)
        layout.addWidget(slider)
        return layout, label

    # ==========================================
    # 【热重载响应函数】下发参数并触发底层线程倒退运算
    # ==========================================
    def on_param_changed(self, slider, label, name, scale):
        val = slider.value()
        display_val = val / scale if scale != 1.0 else val
        label.setText(f"{name}: {display_val}")
        
        if self.thread and self.thread.detector:
            if name == "去水雾强度":
                self.thread.detector.defog_strength = val
            elif name == "轮廓提取灰度阈值":
                self.thread.detector.gray_threshold = val
            elif name == "浪形检测切除比例(%)":
                self.thread.detector.trim_ratio = display_val / 100.0 
            elif name == "浪形检测报警阈值":
                self.thread.detector.wave_amplitude_threshold = display_val
            
            # 若系统挂起，强制触发当前帧的重新渲染
            if self.thread.is_paused:
                self.thread.refresh_current_frame()

    def open_video_dialog(self):
        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(self, "选择监控视频", "", "视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*)", options=options)
        
        if video_path:
            self.video_label.setText("")
            self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
            self.status_label.setText("当前状态：初始化算法...")
            self.log_list.clear() 
            self.frame_label.setText("0 / 0")
            
            if self.thread is not None and self.thread.isRunning():
                self.thread.stop()

            self.thread = VideoThread(video_path)
            
            # 初始化注入前端 UI 控制台参数
            self.on_param_changed(self.sl_defog.itemAt(1).widget(), self.lbl_defog, "去水雾强度", 1.0)
            self.on_param_changed(self.sl_gray.itemAt(1).widget(), self.lbl_gray, "轮廓提取灰度阈值", 1.0)
            self.on_param_changed(self.sl_trim.itemAt(1).widget(), self.lbl_trim, "浪形检测切除比例(%)", 1.0)
            self.on_param_changed(self.sl_amp.itemAt(1).widget(), self.lbl_amp, "浪形检测报警阈值", 10.0)

            # 绑定跨线程通信信号
            self.thread.update_signal.connect(self.update_frame)
            self.thread.init_signal.connect(self.init_video_controls) 
            self.thread.clear_log_signal.connect(self.log_list.clear)
            self.thread.start()

    # ==========================================
    # 【核心模块二】自适应视窗缩放引擎 (Auto-Aspect-Ratio Engine)
    # ==========================================
    def init_video_controls(self, total_frames, w, h):
        self.total_frames = total_frames
        self.slider.setMaximum(total_frames)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.play_button.setText("⏸ 暂停播放")
        self.frame_label.setText(f"0 / {self.total_frames}")

        # 利用源视频长宽比，反向推导并重塑主窗口几何尺寸，消灭画面黑边
        if w > 0 and h > 0:
            video_ratio = w / h

            # 补偿系数 1.04 来源于预设的 QGridLayout 网格权重模型
            current_w = self.width()
            target_h = int(current_w / (video_ratio * 1.04))
            
            # 屏幕溢出安全限制
            screen = QApplication.primaryScreen().availableGeometry()
            max_h = screen.height() - 50
            if target_h > max_h:
                target_h = max_h
                current_w = int(target_h * video_ratio * 1.04)
            
            self.resize(current_w, target_h)

    def toggle_play(self):
        if self.thread:
            self.thread.is_paused = not self.thread.is_paused
            self.play_button.setText("▶ 继续播放" if self.thread.is_paused else "⏸ 暂停播放")

    def slider_pressed(self):
        self.is_slider_dragging = True
        if self.thread:
            self.was_paused_before_drag = self.thread.is_paused
            self.thread.is_paused = True 

    def slider_moved(self, position):
        if self.thread:
            self.thread.set_frame(position)
            self.frame_label.setText(f"{position} / {self.total_frames}")

    def slider_released(self):
        self.is_slider_dragging = False
        if self.thread:
            self.thread.is_paused = self.was_paused_before_drag

    # ==========================================
    # 【核心模块三】历史帧快速定位 (Frame Index Tracing)
    # 响应双击事件，接管视频流实现时光回溯与排障定位
    # ==========================================
    def jump_to_log_frame(self, item):
        frame_idx = item.data(Qt.UserRole) 
        if frame_idx is not None and self.thread:
            self.slider.setValue(frame_idx)
            self.thread.set_frame(frame_idx)
            self.thread.is_paused = True
            self.play_button.setText("▶ 继续播放")

    # ==========================================
    # 【视图渲染】异步接收算法结果并驱动四联屏与日志面板刷新
    # ==========================================
    def update_frame(self, frame, defogged_frame, annotated_frame, heatmap, status, log_flag, current_frame):
        self.display_image(frame, self.video_label)
        self.display_image(defogged_frame, self.defogged_label)
        self.display_image(annotated_frame, self.annotated_label)
        self.display_image(heatmap, self.heatmap_label)
        
        if not self.is_slider_dragging:
            self.slider.blockSignals(True) 
            self.slider.setValue(current_frame)
            self.slider.blockSignals(False)
            self.frame_label.setText(f"{current_frame} / {self.total_frames}")
        
        self.status_label.setText(f"当前状态：{status}")
        if "报警" in status:
            self.status_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #FF3333; background-color: #440000;")
        else:
            self.status_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #00FF00; background-color: transparent;")

        # 结构化日志写入，隐式绑定 UserRole 帧号索引
        if log_flag:
            current_time = datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{current_time}] 第 {current_frame} 帧\n⚠️ 发生异常：{status}"
            
            item = QListWidgetItem(log_msg)
            item.setData(Qt.UserRole, current_frame) 
            
            self.log_list.insertItem(0, item)
            # FIFO 队列维护历史记录上限
            if self.log_list.count() > 100:
                self.log_list.takeItem(self.log_list.count() - 1)

    def display_image(self, img_array, label_widget):
        if len(img_array.shape) == 2:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        # 开启平滑缩放变换以适应自适应拉伸视窗
        scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())