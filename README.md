# 热连轧带钢浪形检测系统

## 项目简介

热连轧带钢浪形检测系统是一个基于Python和计算机视觉的实时检测系统，用于在钢厂热连轧精轧机架间检测带钢的浪形缺陷。该系统能够从工业相机视频流中恢复带钢表面的三维动态轮廓，并自动识别和分类浪形类型，为机架间实时控制提供数据支持。

## 核心功能

1. **视频流接入**：支持工业相机、本地视频和文件夹读图
2. **高温水雾预处理**：去雾、去噪、对比度增强、带钢区域分割
3. **三维轮廓建模**：平面拟合、高度差计算、中线/边线提取
4. **浪形检测分类器**：基于轮廓特征+YOLOv8分类，输出类别+等级
5. **量化测量**：输出浪高、浪宽、位置（mm）
6. **实时Pipeline**：多线程/异步，满足毫秒级指标
7. **可视化**：实时画轮廓、标注浪形、显示测量值
8. **日志+告警+数据上报**：CSV/JSON存结果，支持对接PLC
9. **配置化**：相机参数、标定参数、阈值、IO路径全可配
10. **异常处理**：断流、过曝、黑屏、无带钢自动容错

## 技术指标

- 垂直测量误差＜2mm
- 水平测量误差＜10mm
- 数据输出时延＜500ms
- 系统响应＜200ms
- 无故障率＞99%

## 项目结构

```
├── config/           # 配置文件
│   └── config.yaml   # 主配置文件
├── data/             # 数据目录
│   ├── sample_video.mp4  # 示例视频
│   └── images/       # 示例图像
├── logs/             # 日志目录
├── output/           # 输出目录
├── src/              # 核心源码
│   ├── __init__.py
│   ├── preprocessing.py     # 预处理模块
│   ├── 3d_reconstruction.py # 三维重建模块
│   ├── wave_detector.py     # 浪形检测模块
│   ├── visualization.py     # 可视化模块
│   ├── pipeline.py          # 实时处理Pipeline
│   └── utils.py             # 工具函数
├── tests/            # 测试目录
├── requirements.txt  # 依赖清单
├── run.py            # 启动脚本
└── README.md         # 项目说明
```

## 环境要求

- Python 3.8+
- OpenCV 4.8.0+
- NumPy 1.24.3+
- PyYAML 6.0.1+
- Ultralytics 8.0.196+（用于YOLOv8）
- scikit-learn 1.3.0+
- pandas 2.0.3+
- matplotlib 3.7.2+

## 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 配置说明

配置文件位于 `config/config.yaml`，包含以下主要部分：

- **camera**：相机配置，包括类型、路径、分辨率等
- **calibration**：标定参数，包括像素到毫米的转换系数
- **preprocessing**：预处理配置，包括去雾、去噪、对比度增强等
- **reconstruction**：三维重建配置，包括重建方法、平面拟合等
- **detection**：浪形检测配置，包括浪形类型、等级阈值等
- **realtime**：实时处理配置，包括多线程、异步处理等
- **visualization**：可视化配置，包括显示参数、绘图参数等
- **logging**：日志配置，包括级别、文件路径等
- **output**：输出配置，包括CSV、JSON输出等
- **exception**：异常处理配置，包括断流、过曝、黑屏等

## 运行系统

### 基本运行

```bash
# 使用默认配置运行
python run.py

# 使用指定配置文件
python run.py --config config/config.yaml

# 处理本地视频
python run.py --video data/sample_video.mp4

# 使用工业相机
python run.py --camera 0

# 处理图像文件夹
python run.py --folder data/images

# 开启调试模式
python run.py --debug
```

### 键盘控制

- `q`：退出系统
- `s`：保存当前帧

## 输出说明

- **CSV输出**：`output/detection_results.csv`，包含时间戳、浪形类型、等级、高度、宽度、位置等信息
- **JSON输出**：`output/detection_results.json`，包含详细的检测结果和处理信息
- **日志输出**：`logs/detection.log`，包含系统运行日志
- **图像输出**：`output/capture.jpg`，保存的截图

## 浪形类型

系统支持以下浪形类型：

- **平直**：带钢表面平整，无明显波浪
- **DS单边浪**：驱动侧单边浪
- **WS单边浪**：工作侧单边浪
- **双边浪**：两侧同时出现波浪
- **中浪**：带钢中间出现波浪
- **复合浪形**：同时存在多种浪形

## 浪形等级

系统将浪形分为以下等级：

- **低**：浪高 < 1.0mm
- **中**：1.0mm ≤ 浪高 < 2.0mm
- **高**：2.0mm ≤ 浪高 < 3.0mm
- **严重**：浪高 ≥ 3.0mm

## 性能优化

1. **多线程处理**：使用多线程并行处理帧图像，提高处理速度
2. **队列缓冲**：使用队列缓冲帧数据，避免处理延迟
3. **帧丢弃**：当队列满时，丢弃旧帧，确保实时性
4. **算法优化**：优化预处理和检测算法，减少计算复杂度
5. **内存管理**：合理管理内存，避免内存泄漏

## 异常处理

系统具备以下异常处理能力：

- **断流处理**：相机断流时自动重连
- **过曝处理**：检测到过曝时发出警告
- **黑屏处理**：检测到黑屏时发出警告
- **无带钢处理**：检测到无带钢时发出警告

## 扩展与定制

### 添加新的浪形类型

在 `config/config.yaml` 中修改 `detection.wave_types` 配置，然后在 `src/wave_detector.py` 中实现相应的分类逻辑。

### 调整检测参数

在 `config/config.yaml` 中修改 `detection.parameters` 配置，包括最小浪长、最小浪高等参数。

### 集成新的相机

在 `config/config.yaml` 中修改 `camera` 配置，添加新相机的参数。

## 测试与验证

### 单元测试

```bash
cd tests
python -m pytest test_wave_detection.py -v
```

### 性能测试

```bash
python run.py --video data/sample_video.mp4
```

查看日志文件 `logs/detection.log` 中的性能指标。

## 项目维护

### 日志管理

定期清理 `logs/` 目录下的日志文件，避免占用过多磁盘空间。

### 依赖更新

定期更新 `requirements.txt` 中的依赖包版本，确保系统的稳定性和安全性。

### 模型更新

如果使用YOLOv8模型，定期更新模型文件，提高检测准确率。

## 注意事项

1. **工业环境**：在高温、强水雾环境下，需要确保相机的防护措施到位
2. **标定校准**：定期校准相机参数，确保测量精度
3. **系统维护**：定期检查系统运行状态，确保无故障运行
4. **数据备份**：定期备份检测结果和日志文件

## 联系方式

- 作者：Builder
- 邮箱：builder@example.com
- 项目地址：https://github.com/builder/steel-wave-detection

---

**© 2026 热连轧带钢浪形检测系统**
