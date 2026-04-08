import cv2
import numpy as np
import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.wave_detector import WaveDetector

class WaveDetectionTester:
    def __init__(self):
        self.detector = WaveDetector()
        self.test_results = []
    
    def test_single_video(self, video_path, expected_shape=None):
        """测试单个视频文件"""
        print(f"  打开视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return None
        
        frame_count = 0
        detections = []
        start_time = time.time()
        
        print(f"  开始处理视频...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每10帧检测一次
            if frame_count % 10 == 0:
                shape, height, error = self.detector.detect_from_frame(frame)
                detections.append((shape, height, error))
                if frame_count % 100 == 0:
                    print(f"    帧 {frame_count}: 浪形={shape}, 浪高={height}mm, 偏差={error}mm")
            
            frame_count += 1
        
        cap.release()
        end_time = time.time()
        
        if not detections:
            print(f"  ❌ 未检测到任何有效帧")
            return None
        
        print(f"  处理完成，共检测 {len(detections)} 帧")
        
        # 统计结果
        shape_counts = {}
        for shape, _, _ in detections:
            if shape != "无有效轮廓":
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        print(f"  浪形分布: {shape_counts}")
        
        # 确定主要浪形
        if shape_counts:
            dominant_shape = max(shape_counts, key=shape_counts.get)
            avg_height = np.mean([h for _, h, _ in detections if h > 0])
            avg_error = np.mean([e for _, _, e in detections if e > 0])
        else:
            dominant_shape = "无有效轮廓"
            avg_height = 0.0
            avg_error = 0.0
        
        test_time = end_time - start_time
        fps = frame_count / test_time if test_time > 0 else 0
        
        result = {
            "video": os.path.basename(video_path),
            "expected_shape": expected_shape,
            "detected_shape": dominant_shape,
            "avg_height": round(avg_height, 2),
            "avg_error": round(avg_error, 2),
            "frame_count": frame_count,
            "fps": round(fps, 2),
            "test_time": round(test_time, 2),
            "accuracy": self._calculate_accuracy(dominant_shape, expected_shape)
        }
        
        self.test_results.append(result)
        return result
    
    def _calculate_accuracy(self, detected, expected):
        """计算准确率"""
        if expected is None:
            return 0.5  # 未知情况下给予中等准确率
        
        # 简化的准确率计算
        if detected == expected:
            return 1.0
        elif detected in ["边浪", "中浪"] and expected in ["边浪", "中浪"]:
            return 0.7
        elif detected == "无有效轮廓":
            return 0.2
        else:
            return 0.3
    
    def test_batch(self, video_folder, expected_shapes=None):
        """批量测试视频"""
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(video_folder, video_file)
            expected = expected_shapes[i] if expected_shapes and i < len(expected_shapes) else None
            
            print(f"\n📹 测试视频 {i+1}/{len(video_files)}: {video_file}")
            result = self.test_single_video(video_path, expected)
            
            if result:
                self._print_result(result)
    
    def _print_result(self, result):
        """打印测试结果"""
        print(f"  视频: {result['video']}")
        print(f"  预期浪形: {result['expected_shape']}")
        print(f"  检测浪形: {result['detected_shape']}")
        print(f"  平均浪高: {result['avg_height']}mm")
        print(f"  平均偏差: {result['avg_error']}mm")
        print(f"  帧率: {result['fps']} FPS")
        print(f"  测试时间: {result['test_time']}s")
        print(f"  准确率: {result['accuracy']:.2f}")
    
    def generate_report(self):
        """生成测试报告"""
        if not self.test_results:
            print("❌ 无测试结果")
            return
        
        print("\n" + "="*60)
        print("📊 浪形检测算法测试报告")
        print("="*60)
        
        # 统计总体性能
        total_accuracy = np.mean([r['accuracy'] for r in self.test_results])
        total_fps = np.mean([r['fps'] for r in self.test_results])
        total_time = sum([r['test_time'] for r in self.test_results])
        
        print(f"\n总体统计:")
        print(f"  测试视频数: {len(self.test_results)}")
        print(f"  平均准确率: {total_accuracy:.2f}")
        print(f"  平均帧率: {total_fps:.2f} FPS")
        print(f"  总测试时间: {total_time:.2f}s")
        
        # 浪形分布
        shape_distribution = {}
        for result in self.test_results:
            shape = result['detected_shape']
            shape_distribution[shape] = shape_distribution.get(shape, 0) + 1
        
        print(f"\n浪形检测分布:")
        for shape, count in shape_distribution.items():
            percentage = (count / len(self.test_results)) * 100
            print(f"  {shape}: {count} ({percentage:.1f}%)")
        
        # 性能分析
        print(f"\n性能分析:")
        print(f"  算法处理速度满足实时要求: {'是' if total_fps >= 25 else '否'}")
        print(f"  平均误差控制: {'良好' if np.mean([r['avg_error'] for r in self.test_results]) < 2 else '需要改进'}")
        
        print("\n" + "="*60)
    
    def test_accuracy(self, test_cases):
        """测试特定场景的准确率"""
        print("\n" + "="*60)
        print("🎯 特定场景准确率测试")
        print("="*60)
        
        for test_case, expected_shape in test_cases.items():
            print(f"\n测试场景: {test_case}")
            result = self.test_single_video(test_case, expected_shape)
            if result:
                self._print_result(result)
        
        self.generate_report()

if __name__ == "__main__":
    tester = WaveDetectionTester()
    
    # 测试无浪形视频
    print("\n测试无浪形视频:")
    no_wave_video = "样例视频及识别需求\\无浪形视频.mp4"
    if os.path.exists(no_wave_video):
        tester.test_single_video(no_wave_video, "平直")
    else:
        print(f"❌ 视频文件不存在: {no_wave_video}")
    
    # 测试工作侧有单边浪视频
    print("\n测试工作侧有单边浪视频:")
    edge_wave_video = "样例视频及识别需求\\工作侧有单边浪.mp4"
    if os.path.exists(edge_wave_video):
        tester.test_single_video(edge_wave_video, "边浪")
    else:
        print(f"❌ 视频文件不存在: {edge_wave_video}")
    
    # 测试严重双边浪视频
    print("\n测试严重双边浪视频:")
    center_wave_video = "样例视频及识别需求\\第一和第二块都有严重双边浪.mp4"
    if os.path.exists(center_wave_video):
        tester.test_single_video(center_wave_video, "中浪")
    else:
        print(f"❌ 视频文件不存在: {center_wave_video}")
    
    # 生成最终报告
    tester.generate_report()