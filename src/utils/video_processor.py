import cv2
import os
import json
from datetime import datetime
from src.wave_detector import WaveDetector

class VideoProcessor:
    def __init__(self):
        self.detector = WaveDetector()
    
    def process_single_video(self, video_path, output_dir=None):
        """处理单个视频文件"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_video = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
            output_json = os.path.join(output_dir, f"analysis_{os.path.splitext(os.path.basename(video_path))[0]}.json")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        else:
            out = None
            output_json = None
        
        frame_count = 0
        results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测浪形
            shape, height, error = self.detector.detect_from_frame(frame)
            analysis = self.detector.analyze_wave_pattern(frame)
            
            # 在帧上绘制信息
            cv2.putText(frame, f"浪形: {shape}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"浪高: {height}mm", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"水平偏差: {error}mm", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"帧: {frame_count}/{total_frames}", (20, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 保存处理后的帧
            if out:
                out.write(frame)
            
            # 记录结果
            result = {
                "frame": frame_count,
                "timestamp": frame_count / fps if fps > 0 else 0,
                "wave_shape": shape,
                "wave_height": height,
                "horizontal_error": error,
                "analysis": analysis
            }
            results.append(result)
            
            frame_count += 1
            
            # 显示处理进度
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # 释放资源
        cap.release()
        if out:
            out.release()
        
        # 保存分析结果
        if output_json:
            analysis_summary = {
                "video_info": {
                    "input_path": video_path,
                    "output_path": output_video if output_dir else "",
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                    "total_frames": total_frames,
                    "processed_frames": frame_count,
                    "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": results,
                "summary": self._generate_summary(results)
            }
            
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(analysis_summary, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 分析结果已保存至: {output_json}")
        
        return results
    
    def process_batch(self, video_folder, output_dir):
        """批量处理视频文件夹"""
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        total_videos = len(video_files)
        
        print(f"开始批量处理 {total_videos} 个视频文件...")
        
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(video_folder, video_file)
            video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
            
            print(f"\n处理视频 {i+1}/{total_videos}: {video_file}")
            self.process_single_video(video_path, video_output_dir)
        
        print(f"\n✅ 批量处理完成！所有结果已保存至: {output_dir}")
    
    def _generate_summary(self, results):
        """生成分析摘要"""
        if not results:
            return {}
        
        # 统计浪形分布
        shape_counts = {}
        heights = []
        errors = []
        
        for result in results:
            shape = result['wave_shape']
            if shape != "无有效轮廓":
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
                heights.append(result['wave_height'])
                errors.append(result['horizontal_error'])
        
        # 计算统计信息
        if heights:
            avg_height = sum(heights) / len(heights)
            max_height = max(heights)
            min_height = min(heights)
        else:
            avg_height = 0
            max_height = 0
            min_height = 0
        
        if errors:
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
        else:
            avg_error = 0
            max_error = 0
        
        # 确定主要浪形
        if shape_counts:
            dominant_shape = max(shape_counts, key=shape_counts.get)
        else:
            dominant_shape = "无有效轮廓"
        
        return {
            "dominant_wave_shape": dominant_shape,
            "shape_distribution": shape_counts,
            "wave_height": {
                "average": round(avg_height, 2),
                "maximum": round(max_height, 2),
                "minimum": round(min_height, 2)
            },
            "horizontal_error": {
                "average": round(avg_error, 2),
                "maximum": round(max_error, 2)
            },
            "total_frames_analyzed": len(results)
        }
    
    def extract_key_frames(self, video_path, output_dir, interval=30):
        """提取关键帧"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return
        
        frame_count = 0
        key_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                key_frame_path = os.path.join(output_dir, f"keyframe_{key_frame_count:04d}.jpg")
                cv2.imwrite(key_frame_path, frame)
                key_frame_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"✅ 提取了 {key_frame_count} 个关键帧，保存至: {output_dir}")