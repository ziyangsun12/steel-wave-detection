import sys
import cv2
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QListWidget, QListWidgetItem,
    QGridLayout, QFileDialog, QSizePolicy, QSlider, QHBoxLayout, 
    QVBoxLayout, QPushButton, QGroupBox, QCheckBox, QSpinBox, QTabWidget,
    QComboBox, QStackedWidget, QMainWindow, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QTime, QTimer, QPoint, QRect, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen, QBrush
from PyQt5.QtCore import QThread
from src.wave_detector import WaveDetector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 主页类
# ==========================================
class HomePage(QWidget):
    mode_selected = pyqtSignal(str)  # 发出模式选择信号
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('带钢动态轮廓提取与浪形检测系统')
        self.resize(800, 600)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        # 标题
        title_label = QLabel('带钢动态轮廓提取与浪形检测系统')
        title_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #00CCFF; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel('工业级实时视频分析系统')
        subtitle_label.setStyleSheet("font-size: 18px; color: #CCCCCC; margin-bottom: 40px;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)
        
        # 按钮区域
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)
        
        # 常规模式按钮
        regular_button = QPushButton('常规模式')
        regular_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #00CCFF;
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #00EEFF;
            }
        """)
        regular_button.setFixedHeight(60)
        regular_button.clicked.connect(lambda: self.mode_selected.emit('regular'))
        button_layout.addWidget(regular_button)
        
        # 调试模式按钮
        debug_button = QPushButton('调试模式')
        debug_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #FF6600;
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #FF8800;
            }
        """)
        debug_button.setFixedHeight(60)
        debug_button.clicked.connect(lambda: self.mode_selected.emit('debug'))
        button_layout.addWidget(debug_button)
        
        # 使用说明按钮
        help_button = QPushButton('使用说明')
        help_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #00CC00;
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #00EE00;
            }
        """)
        help_button.setFixedHeight(60)
        help_button.clicked.connect(lambda: self.mode_selected.emit('help'))
        button_layout.addWidget(help_button)
        
        # 退出按钮
        exit_button = QPushButton('退出系统')
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #FF3333;
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #FF5555;
            }
        """)
        exit_button.setFixedHeight(60)
        exit_button.clicked.connect(self.exit_system)
        button_layout.addWidget(exit_button)
        
        # 添加按钮区域到主布局
        main_layout.addLayout(button_layout, 1)
        
        # 底部信息
        info_label = QLabel('版本 2.0 | 工业级带钢检测系统')
        info_label.setStyleSheet("font-size: 14px; color: #888888;")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)
        
        self.setLayout(main_layout)
    
    def exit_system(self):
        """退出系统"""
        reply = QMessageBox.question(self, '退出系统', '确定要退出系统吗？',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()

# ==========================================
# 使用说明页面
# ==========================================
class HelpPage(QWidget):
    back_to_home = pyqtSignal()  # 返回主页信号
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('使用说明 - 带钢动态轮廓提取与浪形检测系统')
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        
        # 标题
        title_label = QLabel('使用说明')
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00CCFF; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)
        
        # 常规模式说明
        regular_section = QGroupBox('常规模式')
        regular_section.setStyleSheet("QGroupBox { border: 1px solid #666; margin-top: 1ex; font-weight: bold; color: #00CCFF; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        regular_layout = QVBoxLayout()
        regular_text = QLabel('''
常规模式仅展示原始监控画面和状态分析，适合日常监控使用。

功能：
- 显示原始监控视频
- 实时显示带钢状态分析结果
- 支持基本的视频控制（播放/暂停）
''')
        regular_text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        regular_text.setWordWrap(True)
        regular_layout.addWidget(regular_text)
        regular_section.setLayout(regular_layout)
        content_layout.addWidget(regular_section)
        
        # 调试模式说明
        debug_section = QGroupBox('调试模式')
        debug_section.setStyleSheet("QGroupBox { border: 1px solid #666; margin-top: 1ex; font-weight: bold; color: #FF6600; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        debug_layout = QVBoxLayout()
        debug_text = QLabel('''
调试模式提供完整的功能，适合技术人员进行参数调整和系统调试。

功能：
- 显示原始监控视频
- 显示背景物理隔离与去雾效果
- 显示带钢轮廓提取与浪形检测区域标定
- 显示带钢形貌伪色彩渲染
- 实时调整算法参数
- 查看详细的日志记录
- 支持高级的视频控制（帧步进、进度条拖动等）
''')
        debug_text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        debug_text.setWordWrap(True)
        debug_layout.addWidget(debug_text)
        debug_section.setLayout(debug_layout)
        content_layout.addWidget(debug_section)
        
        # 操作说明
        operation_section = QGroupBox('操作说明')
        operation_section.setStyleSheet("QGroupBox { border: 1px solid #666; margin-top: 1ex; font-weight: bold; color: #00CC00; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        operation_layout = QVBoxLayout()
        operation_text = QLabel('''
基本操作：
1. 在主页选择需要的模式
2. 在常规模式或调试模式中，点击视频区域或拖拽视频文件到窗口导入视频
3. 常规模式下，查看实时状态分析结果
4. 调试模式下，可以调整算法参数并查看详细的分析结果
5. 在调试模式下，可以通过日志记录查看历史报警信息
6. 点击退出按钮或关闭窗口退出系统
''')
        operation_text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        operation_text.setWordWrap(True)
        operation_layout.addWidget(operation_text)
        operation_section.setLayout(operation_layout)
        content_layout.addWidget(operation_section)
        
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget, 1)
        
        # 返回按钮
        back_button = QPushButton('返回主页')
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #00CCFF;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #00EEFF;
            }
        """)
        back_button.setFixedHeight(40)
        back_button.clicked.connect(self.back_to_home.emit)
        main_layout.addWidget(back_button)
        
        self.setLayout(main_layout)

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


# ==========================================
# 可拖拽的分隔线包装器
# ==========================================
class DraggableSplitter(QSplitter):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(6)
        self.setStyleSheet("""
            QSplitter::handle {
                background-color: #555555;
            }
            QSplitter::handle:hover {
                background-color: #00CCFF;
            }
            QSplitter::handle:pressed {
                background-color: #FF6600;
            }
        """)

# ==========================================
# 可折叠的侧边栏面板（支持动画）
# ==========================================
class CollapsiblePanel(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.is_expanded = True
        
        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 折叠按钮（侧边栏）
        self.toggle_btn = QPushButton("日\n志\n记\n录\n◄")
        self.toggle_btn.setFixedSize(50, 160)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                color: white;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_panel)
        
        # 内容区域
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(12)
        self.content_widget.setLayout(self.content_layout)
        self.content_widget.setStyleSheet("""
            background-color: #2D2D2D;
            border-left: 1px solid #555555;
        """)
        
        main_layout.addWidget(self.toggle_btn)
        main_layout.addWidget(self.content_widget, 1)
        
        self.setLayout(main_layout)
        # 移除所有固定宽度设置，让 QSplitter 完全控制宽度

    def toggle_panel(self):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.toggle_btn.setText("◄")
            self.content_widget.show()
        else:
            self.toggle_btn.setText("►")
            self.content_widget.hide()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

    def add_stretch(self):
        self.content_layout.addStretch()

class ClickableProgressBar(QSlider):
    """支持点击跳转的进度条"""
    position_clicked = pyqtSignal(int)  # 发出点击位置信号
    
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        # 计算点击位置对应的值
        if self.width() > 0:
            value = int((event.x() / self.width()) * self.maximum())
            self.setValue(value)
            self.position_clicked.emit(value)  # 发出信号
        super().mousePressEvent(event)

# ==========================================
# 进度条带关键帧标记
# ==========================================
class AdvancedProgressBar(QWidget):
    position_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 100
        self.current_frame = 0
        self.alarm_frames = []  # 记录所有报警帧
        self.is_dragging = False
        
        # UI元素
        layout = QVBoxLayout()
        layout.setSpacing(8)  # 增加间距
        layout.setContentsMargins(8, 8, 8, 8)  # 增加边距
        
        # 时间戳显示
        time_layout = QHBoxLayout()
        time_layout.setSpacing(12)
        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet("font-size: 13px; color: #00CCFF; font-weight: bold; min-width: 70px;")
        self.total_time_label = QLabel("00:00:00")
        self.total_time_label.setStyleSheet("font-size: 13px; color: #CCCCCC; min-width: 70px;")
        time_layout.addWidget(QLabel("当前时间："))
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(QLabel("总时长："))
        time_layout.addWidget(self.total_time_label)
        time_layout.addStretch()
        
        # 滑块和进度条
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(8)
        self.slider = ClickableProgressBar(Qt.Horizontal)   
        # 连接信号
        self.slider.position_clicked.connect(self.on_slider_clicked)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 10px;
                background: #444444;
                border-radius: 5px;
                margin: 4px 0px;
            }
            QSlider::handle:horizontal {
                background: #00CCFF;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
                border: 2px solid #0088AA;
            }
            QSlider::handle:horizontal:hover {
                background: #00EEFF;
            }
        """)
        slider_layout.addWidget(self.slider)
        
        # 帧数显示
        frame_layout = QHBoxLayout()
        frame_layout.setSpacing(8)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setStyleSheet("font-size: 12px; color: #CCCCCC; min-width: 100px;")
        frame_layout.addStretch()
        frame_layout.addWidget(self.frame_label)
        
        layout.addLayout(time_layout)
        layout.addLayout(slider_layout)
        layout.addLayout(frame_layout)
        
        self.setLayout(layout)
        
        # 信号连接
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)

    def on_slider_clicked(self, value):
        """处理点击事件"""
        frame_index = int(value * self.total_frames / 100)
        self.position_changed.emit(frame_index)
    
    def on_slider_pressed(self):
        self.is_dragging = True
    
    def on_slider_moved(self, value):
        self.current_frame = int(value * self.total_frames / 100)
        self.update_time_display()
        self.position_changed.emit(self.current_frame)
    
    def on_slider_released(self):
        self.is_dragging = False
    
    def update_frame(self, current, total):
        self.total_frames = total
        self.current_frame = current
        if not self.is_dragging:
            self.slider.setValue(int(current * 100 / max(total, 1)))
        self.frame_label.setText(f"Frame: {current} / {total}")
        self.update_time_display()
        self.update()  # 重绘以显示关键帧标记
    
    def update_time_display(self):
        fps = 30
        current_seconds = self.current_frame / fps
        total_seconds = self.total_frames / fps
        
        current_time = self.format_time(current_seconds)
        total_time = self.format_time(total_seconds)
        
        self.time_label.setText(current_time)
        self.total_time_label.setText(total_time)
    
    @staticmethod
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def add_alarm_frame(self, frame_index):
        """添加报警帧标记"""
        if frame_index not in self.alarm_frames:
            self.alarm_frames.append(frame_index)
            self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.alarm_frames or self.total_frames == 0:
            return
        
        # 绘制进度条上的关键帧标记
        painter = QPainter(self)
        slider_rect = self.slider.geometry()
        
        for alarm_frame in self.alarm_frames:
            x = slider_rect.x() + int(alarm_frame / self.total_frames * slider_rect.width())
            y = slider_rect.y() + slider_rect.height() // 2
            
            # 绘制红色小点
            painter.setBrush(QBrush(QColor(255, 0, 0)))
            painter.drawEllipse(x - 3, y - 8, 6, 16)


# ==========================================
# 日志管理器（支持分类和过滤）
# ==========================================
class LogManager(QWidget):
    item_double_clicked = pyqtSignal(int)  # 发射帧号信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logs = []  # [(timestamp, level, message, frame_index), ...]
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)  # 增加间距
        layout.setContentsMargins(8, 8, 8, 8)  # 增加边距
        
        # 过滤栏
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)
        filter_layout.addWidget(QLabel("日志过滤："))
        
        self.check_info = QCheckBox("ℹ️ INFO")
        self.check_info.setChecked(True)
        self.check_info.setStyleSheet("QCheckBox { padding: 4px; }")
        
        self.check_warn = QCheckBox("⚠️ WARN")
        self.check_warn.setChecked(True)
        self.check_warn.setStyleSheet("QCheckBox { padding: 4px; }")
        
        self.check_alarm = QCheckBox("🚨 ALARM")
        self.check_alarm.setChecked(True)
        self.check_alarm.setStyleSheet("QCheckBox { padding: 4px; }")
        
        self.check_info.stateChanged.connect(self.refresh_display)
        self.check_warn.stateChanged.connect(self.refresh_display)
        self.check_alarm.stateChanged.connect(self.refresh_display)
        
        filter_layout.addWidget(self.check_info)
        filter_layout.addWidget(self.check_warn)
        filter_layout.addWidget(self.check_alarm)
        filter_layout.addStretch()
        
        # 清空按钮
        clear_btn = QPushButton("清空日志")
        clear_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                background-color: #333333;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        clear_btn.clicked.connect(self.clear_logs)
        filter_layout.addWidget(clear_btn)
        
        # 日志列表
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("""
            QListWidget {
                background-color: #2D2D2D;
                border: 1px solid #555;
                color: white;
                font-size: 11px;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 6px;
                margin: 2px 0px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:hover {
                background-color: #3A3A3A;
            }
            QListWidget::item:selected {
                background-color: #555555;
                border-radius: 2px;
            }
        """)
        self.log_list.itemDoubleClicked.connect(self.on_log_item_clicked)
        
        layout.addLayout(filter_layout)
        layout.addWidget(self.log_list)
        
        self.setLayout(layout)
    
    def add_log(self, level, message, frame_index):
        """
        添加日志
        level: 'INFO', 'WARN', 'ALARM'
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append((timestamp, level, message, frame_index))
        
        # 只保留最近1000条
        if len(self.logs) > 1000:
            self.logs.pop(0)
        
        self.refresh_display()
    
    def refresh_display(self):
        """根据过滤条件刷新显示"""
        self.log_list.clear()
        
        show_info = self.check_info.isChecked()
        show_warn = self.check_warn.isChecked()
        show_alarm = self.check_alarm.isChecked()
        
        # 反向迭代，最新的日志在最上面
        for timestamp, level, message, frame_index in reversed(self.logs):
            if level == 'INFO' and not show_info:
                continue
            if level == 'WARN' and not show_warn:
                continue
            if level == 'ALARM' and not show_alarm:
                continue
            
            icon = self.get_level_icon(level)
            display_text = f"{icon} [{timestamp}] {message}"
            
            item = QListWidgetItem(display_text)
            # 存储帧号作为用户数据
            item.setData(Qt.UserRole, frame_index)
            
            # 设置颜色
            if level == 'ALARM':
                item.setForeground(QColor(255, 0, 0))
            elif level == 'WARN':
                item.setForeground(QColor(255, 165, 0))
            else:
                item.setForeground(QColor(100, 200, 100))
            
            self.log_list.addItem(item)
    
    def on_log_item_clicked(self, item):
        """双击日志项时发射帧号信号"""
        frame_index = item.data(Qt.UserRole)
        self.item_double_clicked.emit(frame_index)
    
    def clear_logs(self):
        """清空所有日志"""
        self.logs.clear()
        self.log_list.clear()
    
    @staticmethod
    def get_level_icon(level):
        if level == 'ALARM':
            return "🚨"
        elif level == 'WARN':
            return "⚠️"
        else:
            return "ℹ️"



# ==========================================
# 常规模式窗口
# ==========================================
class RegularModeWindow(QWidget):
    back_to_home = pyqtSignal()  # 返回主页信号
    switch_to_debug = pyqtSignal()  # 切换到调试模式信号
    
    def __init__(self):
        super().__init__()
        self.thread = None 
        self.is_slider_dragging = False
        self.was_paused_before_drag = False
        self.total_frames = 0
        self.setAcceptDrops(True)
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('带钢动态轮廓提取与浪形检测系统 (常规模式)')
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 标题
        title_label = QLabel('常规模式 - 带钢状态监控')
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00CCFF; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 视频显示区域
        video_layout = QVBoxLayout()
        video_layout.setSpacing(10)
        
        # 原始监控视频
        self.video_label = ClickableLabel("点击此处\n导入现场监控视频")
        self.setup_label_style(self.video_label, dashed=True)
        self.video_label.clicked.connect(self.open_video_dialog)
        video_layout.addWidget(QLabel("原始监控视频"))
        video_layout.addWidget(self.video_label)
        video_layout.setStretch(0, 1) # 标题占 1
        video_layout.setStretch(1, 10) # 视频占 10，保证其尽可能撑满屏幕
        
        # 状态显示
        self.status_label = QLabel("当前状态：请先导入视频")
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(50)
        video_layout.addWidget(self.status_label)
        
        # 视频控制
        video_control_layout = QHBoxLayout()
        video_control_layout.setSpacing(10)
        
        self.play_button = QPushButton("▶ 播放/暂停")
        self.play_button.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        self.play_button.setFixedSize(120, 35)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False) 
        
        # 简单进度条
        self.progress_bar = AdvancedProgressBar()
        self.progress_bar.position_changed.connect(self.on_progress_changed)
        
        video_control_layout.addWidget(self.play_button)
        video_control_layout.addWidget(self.progress_bar, 1)
        
        # 参数设置面板
        param_group = QGroupBox("算法参数设置")
        param_group.setStyleSheet("QGroupBox { border: 1px solid #666; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        param_layout = QHBoxLayout()

        self.sl_defog, self.lbl_defog = self.create_param_slider("去水雾强度", 0, 30, 5)
        self.sl_gray, self.lbl_gray = self.create_param_slider("轮廓提取灰度阈值", 100, 255, 150)
        self.sl_trim, self.lbl_trim = self.create_param_slider("浪形检测切除比例(%)", 0, 30, 15, scale=1.0)
        self.sl_amp, self.lbl_amp = self.create_param_slider("浪形检测报警阈值", 10, 150, 45, scale=10.0)

        param_layout.addLayout(self.sl_defog)
        param_layout.addLayout(self.sl_gray)
        param_layout.addLayout(self.sl_trim)
        param_layout.addLayout(self.sl_amp)
        param_group.setLayout(param_layout)
        
        video_layout.addLayout(video_control_layout)
        video_layout.addWidget(param_group)
        main_layout.addLayout(video_layout, 1)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(20, 10, 20, 10)
        
        # 切换到调试模式按钮
        debug_mode_button = QPushButton('切换到调试模式')
        debug_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #FF6600;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #FF8800;
            }
        """)
        debug_mode_button.setFixedHeight(40)
        debug_mode_button.clicked.connect(lambda: self.switch_to_debug.emit())
        
        # 返回主页按钮
        back_button = QPushButton('← 返回主页')
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #00CCFF;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #00EEFF;
            }
        """)
        back_button.setFixedHeight(40)
        back_button.clicked.connect(self.back_to_home.emit)
        
        # 添加按钮到左侧
        bottom_layout.addWidget(debug_mode_button)
        bottom_layout.addWidget(back_button)
        bottom_layout.addStretch()  # 将按钮推到左侧
        
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)
    
    def setup_label_style(self, label, dashed=False):
        label.setAlignment(Qt.AlignCenter)
        border = "2px dashed #555;" if dashed else "1px solid #333;"
        label.setStyleSheet(f"background-color: #000000; border: {border}; color: #888; font-size: 16px;")
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    
    def open_video_dialog(self):
        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(self, "选择监控视频", "", "视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*)", options=options)
        
        if video_path:
            self.video_label.setText("")
            self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
            self.status_label.setText("当前状态：初始化算法...")
            
            if self.thread is not None and self.thread.isRunning():
                self.thread.stop()

            self.thread = VideoThread(video_path)
            
            self.thread.update_signal.connect(self.update_frame)
            self.thread.start()

            # 启用控制按钮
            self.play_button.setEnabled(True)
            self.play_button.setText("⏸ 暂停播放")

    def toggle_play(self):
        # 切换播放/暂停状态
        current_text = self.play_button.text()
        if current_text == "▶ 播放/暂停" or current_text == "▶ 继续播放":
            # 开始播放或继续播放
            self.play_button.setText("⏸ 暂停播放")
            if self.thread and hasattr(self.thread, 'resume'):
                self.thread.resume()
        elif current_text == "⏸ 暂停播放":
            # 暂停播放
            self.play_button.setText("▶ 继续播放")
            if self.thread and hasattr(self.thread, 'pause'):
                self.thread.pause()
    
    def on_progress_changed(self, frame_index):
        """处理进度条变化"""
        if self.thread:
            self.thread.set_frame(frame_index)
    
    def update_frame(self, frame, dehazed_frame, contour_frame, heatmap, status, log_flag, current_frame, total_frames, algorithm_time, wave_height, wave_width, wave_level, edge_data):
        """更新帧显示"""
        self.display_image(frame, self.video_label)
        
        # 更新进度条
        self.progress_bar.update_frame(current_frame, total_frames)
        
        # 更新状态
        status_text = f"当前状态：{status} | 算法时间：{algorithm_time:.2f}ms"
        if wave_height > 0:
            status_text += f" | 浪高：{wave_height:.2f}mm"
        self.status_label.setText(status_text)
        
        if "报警" in status:
            self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF3333; background-color: #440000;")
            # 添加报警帧标记
            self.progress_bar.add_alarm_frame(current_frame)
        else:
            self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00FF00; background-color: transparent;")
    
    def display_image(self, img_array, label_widget):
        if len(img_array.shape) == 2:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)
    
    def dragEnterEvent(self, event):
        """处理拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """处理拖拽释放事件"""
        for url in event.mimeData().urls():
            video_path = url.toLocalFile()
            if video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.video_label.setText("")
                self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
                self.status_label.setText("当前状态：初始化算法...")
                
                if self.thread is not None and self.thread.isRunning():
                    self.thread.stop()

                self.thread = VideoThread(video_path)
                
                self.thread.update_signal.connect(self.update_frame)
                self.thread.start()

                # 启用控制按钮
                self.play_button.setEnabled(True)
                self.play_button.setText("⏸ 暂停播放")
                break

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        event.accept()
    
    def create_param_slider(self, name, min_val, max_val, default_val, scale=1.0):
        layout = QVBoxLayout()
        label = QLabel(f"{name}: {default_val/scale if scale != 1.0 else default_val}")
        label.setAlignment(Qt.AlignCenter)
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        
        slider.valueChanged.connect(lambda val: self.on_param_changed(slider, label, name, scale))
        
        layout.addWidget(label)
        layout.addWidget(slider)
        return layout, label

    def on_param_changed(self, slider, label, name, scale):
        val = slider.value()
        display_val = val / scale if scale != 1.0 else val
        label.setText(f"{name}: {display_val}")
        
        if self.thread and hasattr(self.thread, 'detector'):
            if name == "去水雾强度":
                self.thread.detector.defog_strength = val
            elif name == "轮廓提取灰度阈值":
                self.thread.detector.gray_threshold = val
            elif name == "浪形检测切除比例(%)":
                self.thread.detector.trim_ratio = display_val / 100.0 
            elif name == "浪形检测报警阈值":
                self.thread.detector.wave_amplitude_threshold = display_val

# ==========================================
# 视频线程（增强版）
# ==========================================
class VideoThread(QThread):
    update_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool, int, int, float, float, float, str, dict)
    log_signal = pyqtSignal(str, str, int)  # level, message, frame_index

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_running = True
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.low_resolution = False
        self.detector = WaveDetector()
        self.last_status = "平直"
        self.seek_frame = -1  # 帧位置调整标志位

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        cap = self.cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 如果设置了起始帧，在此处应用
        if self.current_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        while cap.isOpened() and self.is_running:
            # 检查是否需要调整帧位置
            if self.seek_frame != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                self.current_frame = self.seek_frame
                self.seek_frame = -1  # 重置标志位
                continue
            
            # 检查是否暂停
            while self.is_paused and self.is_running:
                self.msleep(100)
                continue
            
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                continue
            
            original_frame = frame.copy()
            if self.low_resolution:
                frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            
            heatmap, status, contour_frame, dehazed_frame = self.detector.process_frame(frame)
            algorithm_time = 0.0
            wave_height = 0.0
            wave_width = 0.0
            wave_level = ""
            edge_data = {}
            
            if self.low_resolution:
                heatmap = cv2.resize(heatmap, (original_frame.shape[1], original_frame.shape[0]))
                dehazed_frame = cv2.resize(dehazed_frame, (original_frame.shape[1], original_frame.shape[0]))
                contour_frame = cv2.resize(contour_frame, (original_frame.shape[1], original_frame.shape[0]))
                frame = original_frame
            
            # 日志记录
            log_flag = False
            log_level = 'INFO'
            log_msg = ""
            
            if "报警" in status and status != self.last_status:
                log_flag = True
                log_level = 'ALARM'
                log_msg = f"检测到浪形报警：{status}"
                self.log_signal.emit(log_level, log_msg, self.current_frame)
            elif status != self.last_status and "平直" in status:
                log_flag = True
                log_level = 'INFO'
                log_msg = f"带钢状态恢复：{status}"
                self.log_signal.emit(log_level, log_msg, self.current_frame)
            
            self.last_status = status
            self.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            self.update_signal.emit(frame, dehazed_frame, contour_frame, heatmap, status, log_flag, 
                                   self.current_frame, self.total_frames, algorithm_time, 
                                   wave_height, wave_width, wave_level, edge_data)
            
            self.msleep(30)

        if hasattr(self, 'cap'):
            self.cap.release()
    
    def stop(self):
        self.is_running = False
        self.wait()
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
    
    def set_frame(self, frame_index):
        """设置起始帧位置（在线程启动前调用）"""
        self.current_frame = frame_index
        # 通过标志位通知视频线程调整帧位置，避免线程冲突
        self.seek_frame = frame_index
    
    def set_low_resolution(self, low_res):
        self.low_resolution = low_res
    
    def update_parameters(self, omega, t0, binary_threshold):
        if hasattr(self, 'detector'):
            self.detector.set_parameters(omega, t0, binary_threshold)

class MainWindow(QWidget):
    back_to_home = pyqtSignal()  # 返回主页信号
    switch_to_regular = pyqtSignal()  # 切换到常规模式信号
    
    def __init__(self):
        super().__init__()
        self.thread = None 
        self.is_slider_dragging = False
        self.was_paused_before_drag = False
        self.total_frames = 0
        self.setAcceptDrops(True)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('带钢动态轮廓提取与浪形检测系统 (工业调参版)')
        self.resize(1500, 950)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")

        self.video_label = ClickableLabel("点击此处\n导入现场监控视频")
        self.setup_label_style(self.video_label, dashed=True)
        self.video_label.clicked.connect(self.open_video_dialog)
        
        self.defogged_label = QLabel("等待视频接入...")
        self.setup_label_style(self.defogged_label)

        self.annotated_label = QLabel("等待视频接入...")
        self.setup_label_style(self.annotated_label)

        self.heatmap_label = QLabel("等待视频接入...")
        self.setup_label_style(self.heatmap_label)

        # 日志管理器
        self.log_manager = LogManager()
        self.log_manager.item_double_clicked.connect(self.on_log_item_clicked)

        # 可折叠的侧边栏
        self.sidebar = CollapsiblePanel("日志记录")
        self.sidebar.add_widget(self.log_manager)

        self.status_label = QLabel("当前状态：请先导入视频")
        self.status_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #888;")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.play_button = QPushButton("▶ 播放/暂停")
        self.play_button.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        self.play_button.setFixedSize(120, 35)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False) 

        # 高级进度条
        self.progress_bar = AdvancedProgressBar()
        self.progress_bar.position_changed.connect(self.on_progress_changed)

        # 帧控制按钮
        self.prev_button = QPushButton("◀ 上一帧")
        self.prev_button.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        self.prev_button.setFixedSize(80, 35)
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("下一帧 ▶")
        self.next_button.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        self.next_button.setFixedSize(80, 35)
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)

        video_control_layout = QHBoxLayout()
        video_control_layout.addWidget(self.play_button)
        video_control_layout.addWidget(self.prev_button)
        video_control_layout.addWidget(self.next_button)
        video_control_layout.addWidget(self.progress_bar, 1)

        param_group = QGroupBox("算法参数实时整定控制台")
        param_group.setStyleSheet("QGroupBox { border: 1px solid #666; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        param_layout = QHBoxLayout()

        self.sl_defog, self.lbl_defog = self.create_param_slider("去水雾强度", 0, 30, 5)
        self.sl_gray, self.lbl_gray = self.create_param_slider("轮廓提取灰度阈值", 100, 255, 200)
        self.sl_trim, self.lbl_trim = self.create_param_slider("浪形检测切除比例(%)", 0, 30, 15, scale=1.0)
        self.sl_amp, self.lbl_amp = self.create_param_slider("浪形检测报警阈值", 10, 150, 50, scale=10.0)

        param_layout.addLayout(self.sl_defog)
        param_layout.addLayout(self.sl_gray)
        param_layout.addLayout(self.sl_trim)
        param_layout.addLayout(self.sl_amp)
        param_group.setLayout(param_layout)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 左侧内容区域
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 视频显示网格
        grid = QGridLayout()
        
        grid.addWidget(QLabel("原始监控视频"), 0, 0)
        grid.addWidget(QLabel("背景物理隔离与去雾效果"), 0, 1)
        
        grid.addWidget(self.video_label, 1, 0)
        grid.addWidget(self.defogged_label, 1, 1)
        
        grid.addWidget(QLabel("带钢轮廓提取与浪形检测区域标定"), 2, 0)
        grid.addWidget(QLabel("带钢形貌伪色彩渲染"), 2, 1)
        
        grid.addWidget(self.annotated_label, 3, 0)
        grid.addWidget(self.heatmap_label, 3, 1)

        grid.setColumnStretch(0, 5) 
        grid.setColumnStretch(1, 5) 
        
        grid.setRowStretch(0, 1) 
        grid.setRowStretch(1, 10) 
        grid.setRowStretch(2, 1)
        grid.setRowStretch(3, 10) 

        left_layout.addLayout(grid)
        left_layout.addWidget(self.status_label)
        left_layout.addLayout(video_control_layout)
        left_layout.addWidget(param_group)

        # 使用可拖拽的分隔线
        splitter = DraggableSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget := QWidget())
        left_widget.setLayout(left_layout)
        splitter.addWidget(self.sidebar)
        # 移除拉伸因子，让分隔线可以自由移动
        # splitter.setStretchFactor(0, 5)
        # splitter.setStretchFactor(1, 1)
        # 设置初始宽度，让左侧占大部分空间
        splitter.setSizes([self.width() - 350, 350])

        # 底部工具栏
        bottom_toolbar = QHBoxLayout()
        bottom_toolbar.setContentsMargins(20, 10, 20, 10)
        
        # 切换到常规模式按钮
        regular_mode_button = QPushButton('切换到常规模式')
        regular_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #00CC00;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #00EE00;
            }
        """)
        regular_mode_button.setFixedHeight(40)
        regular_mode_button.clicked.connect(lambda: self.switch_to_regular.emit())
        
        # 返回主页按钮
        back_button = QPushButton('← 返回主页')
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 2px solid #00CCFF;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border-color: #00EEFF;
            }
        """)
        back_button.setFixedHeight(40)
        back_button.clicked.connect(self.back_to_home.emit)
        
        # 直接添加到主布局，而不是侧边栏
        bottom_toolbar.addWidget(regular_mode_button)
        bottom_toolbar.addWidget(back_button)
        bottom_toolbar.addStretch()  # 将按钮推到左侧

        main_layout.addWidget(splitter)
        main_layout.addLayout(bottom_toolbar)

        self.setLayout(main_layout)

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
        
        slider.valueChanged.connect(lambda val: self.on_param_changed(slider, label, name, scale))
        
        layout.addWidget(label)
        layout.addWidget(slider)
        return layout, label

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

    def open_video_dialog(self):
        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(self, "选择监控视频", "", "视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*)", options=options)
        
        if video_path:
            self.video_label.setText("")
            self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
            self.status_label.setText("当前状态：初始化算法...")
            self.log_manager.clear_logs()
            
            if self.thread is not None and self.thread.isRunning():
                self.thread.stop()

            self.thread = VideoThread(video_path)
            
            self.on_param_changed(self.sl_defog.itemAt(1).widget(), self.lbl_defog, "去水雾强度", 1.0)
            self.on_param_changed(self.sl_gray.itemAt(1).widget(), self.lbl_gray, "轮廓提取灰度阈值", 1.0)
            self.on_param_changed(self.sl_trim.itemAt(1).widget(), self.lbl_trim, "浪形检测切除比例(%)", 1.0)
            self.on_param_changed(self.sl_amp.itemAt(1).widget(), self.lbl_amp, "浪形检测报警阈值", 10.0)

            self.thread.update_signal.connect(self.update_frame)
            self.thread.log_signal.connect(self.log_manager.add_log)
            self.thread.start()

            # 启用控制按钮
            self.play_button.setEnabled(True)
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            self.play_button.setText("⏸ 暂停播放")

    def toggle_play(self):
        # 切换播放/暂停状态
        current_text = self.play_button.text()
        if current_text == "▶ 播放/暂停" or current_text == "▶ 继续播放":
            # 开始播放或继续播放
            self.play_button.setText("⏸ 暂停播放")
            if self.thread and hasattr(self.thread, 'resume'):
                self.thread.resume()
        elif current_text == "⏸ 暂停播放":
            # 暂停播放
            self.play_button.setText("▶ 继续播放")
            if self.thread and hasattr(self.thread, 'pause'):
                self.thread.pause()

    def on_progress_changed(self, frame_index):
        """处理进度条变化"""
        if self.thread:
            self.thread.set_frame(frame_index)

    def prev_frame(self):
        """上一帧"""
        if self.thread:
            next_frame = max(0, self.thread.current_frame - 1)
            self.thread.set_frame(next_frame)

    def next_frame(self):
        """下一帧"""
        if self.thread:
            next_frame = min(self.thread.total_frames - 1, self.thread.current_frame + 1)
            self.thread.set_frame(next_frame)

    def on_log_item_clicked(self, frame_index):
        """双击日志时跳转到对应帧"""
        if self.thread:
            self.thread.set_frame(frame_index)

    def update_frame(self, frame, dehazed_frame, contour_frame, heatmap, status, log_flag, current_frame, total_frames, algorithm_time, wave_height, wave_width, wave_level, edge_data):
        """更新帧显示"""
        self.display_image(frame, self.video_label)
        self.display_image(dehazed_frame, self.defogged_label)
        self.display_image(contour_frame, self.annotated_label)
        self.display_image(heatmap, self.heatmap_label)
        
        # 更新进度条
        self.progress_bar.update_frame(current_frame, total_frames)
        
        # 更新状态
        status_text = f"当前状态：{status} | 算法时间：{algorithm_time:.2f}ms"
        if wave_height > 0:
            status_text += f" | 浪高：{wave_height:.2f}mm"
        self.status_label.setText(status_text)
        
        if "报警" in status:
            self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF3333; background-color: #440000;")
            # 添加报警帧标记
            self.progress_bar.add_alarm_frame(current_frame)
        else:
            self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00FF00; background-color: transparent;")

        # 更新边缘曲线
        
    def update_single_frame(self):
        """更新单帧显示（用于暂停时）"""
        if self.thread is None:
            return
        
        # 手动执行一帧处理并显示
        if hasattr(self.thread, 'cap') and self.thread.cap and self.thread.cap.isOpened():
            # 保存当前帧位置
            current_frame = self.thread.current_frame
            # 设置到指定帧位置
            self.thread.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.thread.cap.read()
            if ret:
                heatmap, status, contour_frame, dehazed_frame = self.thread.detector.process_frame(frame)
                algorithm_time = 0.0
                wave_height = 0.0
                wave_width = 0.0
                wave_level = ""
                edge_data = {}
                self.display_image(frame, self.video_label)
                self.display_image(dehazed_frame, self.defogged_label)
                self.display_image(contour_frame, self.annotated_label)
                self.display_image(heatmap, self.heatmap_label)
                
                # 更新状态
                status_text = f"当前状态：{status} | 算法时间：{algorithm_time:.2f}ms"
                if wave_height > 0:
                    status_text += f" | 浪高：{wave_height:.2f}mm"
                self.status_label.setText(status_text)
                
                if "报警" in status:
                    self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF3333; background-color: #440000;")
                    # 添加报警帧标记
                    self.progress_bar.add_alarm_frame(current_frame)
                else:
                    self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00FF00; background-color: transparent;")
                
                
                # 重置帧位置，避免read()导致的前进
                self.thread.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                # 更新进度条显示
                self.progress_bar.update_frame(current_frame, self.thread.total_frames)

    def display_image(self, img_array, label_widget):
        if len(img_array.shape) == 2:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)

    def dragEnterEvent(self, event):
        """处理拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """处理拖拽释放事件"""
        for url in event.mimeData().urls():
            video_path = url.toLocalFile()
            if video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.video_label.setText("")
                self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #555;")
                self.status_label.setText("当前状态：初始化算法...")
                self.log_manager.clear_logs()
                
                if self.thread is not None and self.thread.isRunning():
                    self.thread.stop()

                self.thread = VideoThread(video_path)
                
                self.on_param_changed(self.sl_defog.itemAt(1).widget(), self.lbl_defog, "去水雾强度", 1.0)
                self.on_param_changed(self.sl_gray.itemAt(1).widget(), self.lbl_gray, "轮廓提取灰度阈值", 1.0)
                self.on_param_changed(self.sl_trim.itemAt(1).widget(), self.lbl_trim, "浪形检测切除比例(%)", 1.0)
                self.on_param_changed(self.sl_amp.itemAt(1).widget(), self.lbl_amp, "浪形检测报警阈值", 10.0)

                self.thread.update_signal.connect(self.update_frame)
                self.thread.log_signal.connect(self.log_manager.add_log)
                self.thread.start()

                # 启用控制按钮
                self.play_button.setEnabled(True)
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                self.play_button.setText("⏸ 暂停播放")
                break

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        event.accept()

# ==========================================
# 主应用类
# ==========================================
class MainApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.current_window = None
        self.show_home_page()
    
    def show_home_page(self):
        """显示主页"""
        if self.current_window:
            self.current_window.close()
        
        self.current_window = HomePage()
        self.current_window.mode_selected.connect(self.handle_mode_selection)
        self.current_window.show()
    
    def handle_mode_selection(self, mode):
        """处理模式选择"""
        if mode == 'regular':
            self.show_regular_mode()
        elif mode == 'debug':
            self.show_debug_mode()
        elif mode == 'help':
            self.show_help_page()
    
    def show_regular_mode(self):
        """显示常规模式"""
        if self.current_window:
            self.current_window.close()
        
        self.current_window = RegularModeWindow()
        self.current_window.back_to_home.connect(self.show_home_page)
        self.current_window.switch_to_debug.connect(self.show_debug_mode)
        self.current_window.show()
    
    def show_debug_mode(self):
        """显示调试模式"""
        if self.current_window:
            self.current_window.close()
        
        self.current_window = MainWindow()
        self.current_window.back_to_home.connect(self.show_home_page)
        self.current_window.switch_to_regular.connect(self.show_regular_mode)
        self.current_window.show()
    
    def show_help_page(self):
        """显示使用说明页面"""
        if self.current_window:
            self.current_window.close()
        
        self.current_window = HelpPage()
        self.current_window.back_to_home.connect(self.show_home_page)
        self.current_window.show()

if __name__ == '__main__':
    app = MainApp(sys.argv)
    sys.exit(app.exec_())