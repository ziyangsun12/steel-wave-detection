import sys
import cv2
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextBrowser
from PyQt6.QtGui import QImage, QPixmap
from src.wave_detector import WaveDetector

class SteelWaveDetectApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("热连轧带钢浪形检测系统")
        self.setFixedSize(900, 650)
        
        # 初始化检测器
        self.detector = WaveDetector()
        
        # ================== 界面组件 ==================
        self.labelVideo = QLabel()
        self.labelVideo.setStyleSheet("background-color: #1a1a1a;")
        self.labelVideo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelVideo.setMinimumHeight(400)
        
        self.btnOpen = QPushButton("打开视频")
        self.btnDetect = QPushButton("开始检测")
        self.btnScreenshot = QPushButton("保存截图")
        
        self.textBrowser = QTextBrowser()
        self.textBrowser.setMaximumHeight(150)
        
        layout = QVBoxLayout()
        layout.addWidget(self.labelVideo)
        
        # 按钮布局
        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.btnOpen)
        btn_layout.addWidget(self.btnDetect)
        btn_layout.addWidget(self.btnScreenshot)
        
        layout.addLayout(btn_layout)
        layout.addWidget(self.textBrowser)
        
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        
        # ================== 检测变量 ==================
        self.cap = None
        self.timer = QTimer()
        self.is_detect = False
        self.frame_count = 0
        
        # ================== 信号绑定 ==================
        self.btnOpen.clicked.connect(self.openVideo)
        self.btnDetect.clicked.connect(self.toggleDetection)
        self.btnScreenshot.clicked.connect(self.saveScreenshot)
        self.timer.timeout.connect(self.processFrame)
        
        self.log("✅ 系统启动成功，等待加载带钢视频...")
        self.log("📌 支持功能：浪形识别、轮廓检测、精度输出")
    
    def log(self, msg):
        """日志输出"""
        self.textBrowser.append(msg)
    
    def openVideo(self):
        """打开视频文件"""
        path, _ = QFileDialog.getOpenFileName(
            filter="视频文件 (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        
        # 释放之前的视频
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.log("❌ 视频打开失败！")
            return
        
        self.log(f"✅ 已加载视频：{path}")
        self.frame_count = 0
    
    def toggleDetection(self):
        """切换检测状态"""
        if self.cap is None or not self.cap.isOpened():
            self.log("⚠️ 请先打开视频！")
            return
        
        self.is_detect = not self.is_detect
        if self.is_detect:
            self.btnDetect.setText("停止检测")
            self.timer.start(30)  # 约33fps
            self.log("▶️ 开始浪形检测...")
        else:
            self.btnDetect.setText("开始检测")
            self.timer.stop()
            self.log("⏹️ 检测停止")
    
    def processFrame(self):
        """处理每一帧"""
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.log("✅ 视频播放完毕")
            return
        
        self.frame_count += 1
        
        # 浪形检测
        shape, height, h_err = self.detector.detect_from_frame(frame)
        
        # 画面叠加显示
        cv2.putText(frame, f"浪形: {shape}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"浪高: {height}mm", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"水平偏差: {h_err}mm", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"帧: {self.frame_count}", (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 转 Qt 图像显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_line = ch * w
        qt_img = QImage(frame_rgb.data, w, h, bytes_line, QImage.Format.Format_RGB888)
        
        # 保持 aspect ratio 显示
        self.labelVideo.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.labelVideo.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        # 每10帧记录一次日志
        if self.frame_count % 10 == 0:
            info = self.detector.get_wave_info(shape, height, h_err)
            self.log(f"📊 检测结果: {shape}, 浪高: {height}mm")
    
    def saveScreenshot(self):
        """保存当前截图"""
        if self.cap is None or not self.cap.isOpened():
            self.log("⚠️ 请先打开视频！")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.log("❌ 无法获取帧！")
            return
        
        # 再次检测并添加信息
        shape, height, h_err = self.detector.detect_from_frame(frame)
        cv2.putText(frame, f"浪形: {shape}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"浪高: {height}mm", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"水平偏差: {h_err}mm", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存文件
        save_path, _ = QFileDialog.getSaveFileName(
            filter="PNG图像 (*.png);;JPEG图像 (*.jpg)"
        )
        if save_path:
            cv2.imwrite(save_path, frame)
            self.log(f"✅ 截图已保存至: {save_path}")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SteelWaveDetectApp()
    win.show()
    sys.exit(app.exec())