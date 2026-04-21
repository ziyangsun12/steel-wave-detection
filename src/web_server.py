"""Web可视化界面模块"""

import os
import sys
import json
import cv2
import numpy as np
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# 添加当前目录到导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用绝对导入
from src.pipeline import Pipeline
from src.wave_detector import WaveDetector

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'web', 'templates'),
            static_folder=os.path.join(project_root, 'web', 'static'))

# 全局变量
pipeline = None
config = None
cap = None
current_frame_index = 0
total_frames = 0
fps = 0
is_playing = True  # 播放状态

# 缓存最新的检测结果
latest_results = []
results_last_updated = 0

# 缓存最新的帧信息
latest_frame_info = {
    'raw_frame': None,
    'dehazed_frame': None,
    'contour_frame': None,
    'heatmap_frame': None,
    'status': '未检测到带钢',
    'algorithm_time': 0,
    'wave_height': 0,
    'wave_width': 0,
    'wave_level': '无'
}


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_detection():
    """开始检测"""
    global pipeline, config, cap, current_frame_index, total_frames, fps
    
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path:
        # 自动查找data文件夹中的视频文件
        data_folder = os.path.join(project_root, 'data')
        if os.path.exists(data_folder):
            # 递归查找所有视频文件
            video_files = []
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_full_path = os.path.join(root, file)
                        video_files.append(video_full_path)
            
            if video_files:
                # 默认处理第一个视频文件
                video_full_path = video_files[0]
                video_path = video_full_path.replace(project_root, '')
                if video_path.startswith('\\'):
                    video_path = video_path[1:]
            else:
                return jsonify({'status': 'error', 'message': '数据文件夹中未找到视频文件'})
        else:
            return jsonify({'status': 'error', 'message': '请选择视频文件'})
    else:
        # 构建绝对路径
        if os.path.isabs(video_path):
            video_full_path = video_path
        else:
            video_full_path = os.path.join(project_root, video_path)
        
        if not os.path.exists(video_full_path):
            return jsonify({'status': 'error', 'message': '视频文件不存在'})
    
    # 加载配置
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            return jsonify({'status': 'error', 'message': '加载配置失败'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'加载配置失败: {str(e)}'})
    
    # 配置视频路径
    config['camera']['type'] = 'local'
    config['camera']['video_path'] = video_full_path
    # 从视频文件名提取标签
    config['label'] = os.path.splitext(os.path.basename(video_full_path))[0]
    
    # 初始化视频捕获
    try:
        cap = cv2.VideoCapture(video_full_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        current_frame_index = 0
        
        # 初始化Pipeline
        pipeline = Pipeline(config)
        pipeline.start()
        
        return jsonify({'status': 'success', 'message': '检测已开始', 'video_path': video_path, 'label': config['label'], 'total_frames': total_frames, 'fps': fps})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'启动失败: {str(e)}'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """停止检测"""
    global pipeline, cap
    
    if pipeline:
        try:
            pipeline.stop()
            pipeline = None
            
            # 释放视频捕获对象
            if cap:
                cap.release()
                cap = None
            
            return jsonify({'status': 'success', 'message': '检测已停止'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'停止失败: {str(e)}'})
    else:
        return jsonify({'status': 'error', 'message': '检测未启动'})


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取检测状态"""
    global pipeline
    
    if pipeline:
        status = pipeline.get_status()
        return jsonify({'status': 'success', 'data': status})
    else:
        return jsonify({'status': 'error', 'message': '检测未启动'})


@app.route('/api/results', methods=['GET'])
def get_results():
    """获取检测结果"""
    global latest_results, results_last_updated
    import time
    
    # 检查是否需要更新结果（每2秒更新一次）
    current_time = time.time()
    if current_time - results_last_updated > 2:
        # 读取CSV结果文件
        csv_file = os.path.join(project_root, 'output', 'detection_results.csv')
        if os.path.exists(csv_file):
            try:
                import pandas as pd
                # 检查文件是否为空
                if os.path.getsize(csv_file) == 0:
                    latest_results = []
                    results_last_updated = current_time
                    return jsonify({'status': 'success', 'data': []})
                
                df = pd.read_csv(csv_file)
                results = df.to_dict('records')
                # 只缓存最新的50条结果
                latest_results = results[-50:]
                results_last_updated = current_time
            except pd.errors.EmptyDataError:
                latest_results = []
                results_last_updated = current_time
                return jsonify({'status': 'success', 'data': []})
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'读取结果失败: {str(e)}'})
        else:
            latest_results = []
            results_last_updated = current_time
            return jsonify({'status': 'success', 'data': []})
    
    return jsonify({'status': 'success', 'data': latest_results})


@app.route('/api/videos', methods=['GET'])
def get_videos():
    """获取所有视频文件"""
    # 递归查找data文件夹下的所有视频文件
    data_folder = os.path.join(project_root, 'data')
    video_files = []
    
    if os.path.exists(data_folder):
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    # 构建相对路径
                    relative_path = os.path.relpath(os.path.join(root, file), project_root)
                    # 转换为正斜杠
                    relative_path = relative_path.replace('\\', '/')
                    video_files.append({
                        'path': relative_path,
                        'name': file,
                        'label': os.path.splitext(file)[0]
                    })
    
    return jsonify({'status': 'success', 'data': video_files})


@app.route('/api/preview', methods=['GET'])
def get_preview():
    """获取视频预览"""
    video_path = request.args.get('video_path')
    
    if not video_path:
        return jsonify({'status': 'error', 'message': '视频文件不存在'})
    
    # 构建绝对路径
    if os.path.isabs(video_path):
        video_full_path = video_path
    else:
        video_full_path = os.path.join(project_root, video_path)
    
    if not os.path.exists(video_full_path):
        return jsonify({'status': 'error', 'message': '视频文件不存在'})
    
    try:
        # 读取视频第一帧
        cap = cv2.VideoCapture(video_full_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'status': 'error', 'message': '读取视频失败'})
        
        # 保存预览图
        preview_path = 'web/static/preview.jpg'
        cv2.imwrite(preview_path, frame)
        
        return jsonify({'status': 'success', 'preview_url': '/static/preview.jpg'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'生成预览失败: {str(e)}'})


@app.route('/api/analyze', methods=['POST'])
def analyze_frame():
    """分析单帧图像"""
    from src.wave_detector import WaveDetector
    
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({'status': 'error', 'message': '图像文件不存在'})
    
    # 构建绝对路径
    if os.path.isabs(image_path):
        image_full_path = image_path
    else:
        image_full_path = os.path.join(project_root, image_path)
    
    if not os.path.exists(image_full_path):
        return jsonify({'status': 'error', 'message': '图像文件不存在'})
    
    try:
        # 加载图像
        frame = cv2.imread(image_full_path)
        if frame is None:
            return jsonify({'status': 'error', 'message': '读取图像失败'})
        
        # 浪形检测
        detector = WaveDetector()
        heatmap, status, algorithm_time, dehazed_frame, contour_frame, wave_height, wave_width, wave_level, edge_data = detector.process_frame(frame)
        
        # 构建结果
        result = {
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
        
        # 保存热力图结果
        vis_path = 'web/static/analysis_result.jpg'
        cv2.imwrite(vis_path, heatmap)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'detect_info': detect_info,
            'visualization_url': '/static/analysis_result.jpg'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'})





@app.route('/api/get_video_info', methods=['GET'])
def get_video_info():
    """获取视频信息"""
    global cap, total_frames, fps, current_frame_index
    
    if not cap:
        return jsonify({'status': 'error', 'message': '检测未启动'})
    
    try:
        # 获取视频信息
        video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'current_frame': current_frame_index
        }
        return jsonify({'status': 'success', 'data': video_info})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取视频信息失败: {str(e)}'})


def generate_frame(frame_type):
    """生成视频帧"""
    global config, current_frame_index, total_frames, latest_frame_info, is_playing
    
    detector = WaveDetector()
    
    while True:
        if not config or not config.get('camera') or not config['camera'].get('video_path'):
            # 发送空帧
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            continue
        
        # 创建新的视频捕获对象
        video_path = config['camera']['video_path']
        cap = cv2.VideoCapture(video_path)
        
        if not cap or not cap.isOpened():
            # 发送空帧
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            continue
        
        # 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        
        if not ret:
            # 视频结束，重置到开始
            current_frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                # 释放视频捕获对象
                cap.release()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
                continue
        
        # 处理帧
        start_time = time.time()
        heatmap, status, algorithm_time, dehazed_frame, contour_frame, wave_height, wave_width, wave_level, edge_data = detector.process_frame(frame)
        process_time = time.time() - start_time
        
        # 更新最新帧信息
        latest_frame_info.update({
            'raw_frame': frame,
            'dehazed_frame': dehazed_frame,
            'contour_frame': contour_frame,
            'heatmap_frame': heatmap,
            'status': status,
            'algorithm_time': algorithm_time,
            'wave_height': wave_height,
            'wave_width': wave_width,
            'wave_level': wave_level
        })
        
        # 根据帧类型选择要发送的帧
        if frame_type == 'raw':
            frame_to_send = frame
        elif frame_type == 'dehazed':
            frame_to_send = dehazed_frame
        elif frame_type == 'contour':
            frame_to_send = contour_frame
        elif frame_type == 'heatmap':
            frame_to_send = heatmap
        else:
            frame_to_send = frame
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        if not ret:
            # 释放视频捕获对象
            cap.release()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            continue
        
        # 发送帧
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 释放视频捕获对象
        cap.release()
        
        # 只有在播放状态下才增加帧索引
        if is_playing:
            current_frame_index = (current_frame_index + 1) % total_frames
        
        # 控制帧率
        time.sleep(0.033)  # 约30fps


@app.route('/video/raw')
def video_raw():
    """原始视频流"""
    return Response(generate_frame('raw'), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/dehazed')
def video_dehazed():
    """去雾后的视频流"""
    return Response(generate_frame('dehazed'), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/contour')
def video_contour():
    """轮廓提取后的视频流"""
    return Response(generate_frame('contour'), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/heatmap')
def video_heatmap():
    """热力图视频流"""
    return Response(generate_frame('heatmap'), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/set_frame', methods=['POST'])
def set_frame():
    """设置视频帧位置"""
    global cap, current_frame_index, total_frames
    
    if not cap:
        return jsonify({'status': 'error', 'message': '检测未启动'})
    
    data = request.json
    frame_index = data.get('frame_index')
    
    if frame_index is None:
        return jsonify({'status': 'error', 'message': '请提供帧索引'})
    
    try:
        # 设置视频帧位置
        frame_index = int(frame_index)
        if 0 <= frame_index < total_frames:
            current_frame_index = frame_index
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
            return jsonify({'status': 'success', 'message': '帧位置已设置', 'current_frame': current_frame_index})
        else:
            return jsonify({'status': 'error', 'message': '帧索引超出范围'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'设置帧位置失败: {str(e)}'})


@app.route('/api/play', methods=['POST'])
def play_video():
    """播放视频"""
    global is_playing
    
    is_playing = True
    return jsonify({'status': 'success', 'message': '视频已开始播放', 'is_playing': is_playing})


@app.route('/api/pause', methods=['POST'])
def pause_video():
    """暂停视频"""
    global is_playing
    
    is_playing = False
    return jsonify({'status': 'success', 'message': '视频已暂停', 'is_playing': is_playing})


@app.route('/api/get_frame_info', methods=['GET'])
def get_frame_info():
    """获取当前帧信息"""
    global latest_frame_info
    
    return jsonify({'status': 'success', 'data': latest_frame_info})


@app.route('/api/update_params', methods=['POST'])
def update_params():
    """更新算法参数"""
    global pipeline
    
    data = request.json
    omega = data.get('omega')
    t0 = data.get('t0')
    binary_threshold = data.get('binary_threshold')
    
    if omega is None or t0 is None or binary_threshold is None:
        return jsonify({'status': 'error', 'message': '请提供完整的参数'})
    
    try:
        # 更新WaveDetector参数
        if pipeline and hasattr(pipeline, 'detector'):
            pipeline.detector.set_parameters(omega, t0, binary_threshold)
        
        # 直接更新全局detector参数
        detector = WaveDetector()
        detector.set_parameters(omega, t0, binary_threshold)
        
        return jsonify({'status': 'success', 'message': '参数已更新', 'params': {'omega': omega, 't0': t0, 'binary_threshold': binary_threshold}})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'更新参数失败: {str(e)}'})


@app.route('/static/<path:path>')
def send_static(path):
    """静态文件服务"""
    return send_from_directory(os.path.join(project_root, 'web', 'static'), path)


def run_web_server(host='0.0.0.0', port=5001):
    """运行Web服务器"""
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    run_web_server()
