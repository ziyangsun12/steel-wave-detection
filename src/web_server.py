"""Web可视化界面模块"""

import os
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.pipeline import Pipeline
from src.utils import Utils

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'web', 'templates'),
            static_folder=os.path.join(project_root, 'web', 'static'))

# 全局变量
pipeline = None
config = None


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_detection():
    """开始检测"""
    global pipeline, config
    
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path:
        return jsonify({'status': 'error', 'message': '请选择视频文件'})
    
    # 构建绝对路径
    if os.path.isabs(video_path):
        video_full_path = video_path
    else:
        video_full_path = os.path.join(project_root, video_path)
    
    if not os.path.exists(video_full_path):
        return jsonify({'status': 'error', 'message': '视频文件不存在'})
    
    # 加载配置
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    config = Utils.load_config(config_path)
    if not config:
        return jsonify({'status': 'error', 'message': '加载配置失败'})
    
    # 配置视频路径
    config['camera']['type'] = 'local'
    config['camera']['video_path'] = video_full_path
    
    # 初始化Pipeline
    try:
        pipeline = Pipeline(config)
        pipeline.start()
        return jsonify({'status': 'success', 'message': '检测已开始'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'启动失败: {str(e)}'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """停止检测"""
    global pipeline
    
    if pipeline:
        try:
            pipeline.stop()
            pipeline = None
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
    # 读取CSV结果文件
    csv_file = os.path.join(project_root, 'output', 'detection_results.csv')
    if os.path.exists(csv_file):
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            results = df.to_dict('records')
            return jsonify({'status': 'success', 'data': results})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'读取结果失败: {str(e)}'})
    else:
        return jsonify({'status': 'error', 'message': '结果文件不存在'})


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
    from src.preprocessing import Preprocessor
    from src.reconstruction_3d import Reconstructor
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
        
        # 加载配置
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        config = Utils.load_config(config_path)
        
        # 预处理
        preprocessor = Preprocessor(config)
        binary, preprocess_info = preprocessor.preprocess(frame)
        
        # 分割带钢
        mask, segment_info = preprocessor.segment_steel(binary)
        
        # 三维重建
        reconstructor = Reconstructor(config)
        contour_data, reconstruct_info = reconstructor.reconstruct(binary, mask)
        
        # 浪形检测
        detector = WaveDetector(config)
        result, detect_info = detector.detect(contour_data)
        
        # 可视化
        from src.visualization import Visualizer
        visualizer = Visualizer(config)
        vis_frame = visualizer.visualize(frame, result, contour_data)
        
        # 保存可视化结果
        vis_path = 'web/static/analysis_result.jpg'
        cv2.imwrite(vis_path, vis_frame)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'preprocess_info': preprocess_info,
            'segment_info': segment_info,
            'reconstruct_info': reconstruct_info,
            'detect_info': detect_info,
            'visualization_url': '/static/analysis_result.jpg'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'})


@app.route('/static/<path:path>')
def send_static(path):
    """静态文件服务"""
    return send_from_directory(os.path.join(project_root, 'web', 'static'), path)


def run_web_server(host='0.0.0.0', port=5000):
    """运行Web服务器"""
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    run_web_server()
