# 带钢动态轮廓提取与浪形检测系统

## 项目概述

带钢动态轮廓提取与浪形检测系统是一款基于Python和计算机视觉的实时检测系统，专为热轧带钢的浪形缺陷检测而设计。系统采用先进的暗通道去雾算法和轮廓提取技术，能够实时监测带钢的浪形缺陷，并提供精确的浪高、浪宽等参数。

系统提供两种运行方式：
- **桌面应用程序**：基于PyQt5开发的可视化界面，支持常规模式和调试模式
- **Web服务器**：基于Flask的Web界面，支持远程访问和实时监控

## 核心功能

### 桌面应用程序功能
1. **常规模式**：仅展示原始监控画面和状态分析，适合日常监控使用
2. **调试模式**：提供完整的功能，包括四路视频显示和详细日志记录
3. **视频控制**：支持播放/暂停、帧步进、进度拖动等操作
4. **参数调整**：实时调整算法参数（去雾强度、灰度阈值、切除比例、报警阈值）
5. **状态显示**：实时显示带钢状态和检测结果
6. **日志记录**：支持INFO、WARN、ALARM等级别的日志过滤和查看

### 图像处理功能
1. **视频流处理**：支持本地视频文件的读取和处理
2. **预处理**：暗通道去雾、降噪、对比度增强
3. **轮廓提取**：带钢区域分割、边缘检测
4. **浪形检测**：基于轮廓特征的浪形识别和分类
5. **量化测量**：输出浪高、浪宽等参数（mm）
6. **实时处理**：多线程处理，满足实时检测需求
7. **可视化**：实时视频流、热力图、结果展示

### Web界面功能
8. **Web界面**：提供直观的Web可视化界面
9. **报警系统**：异常情况实时报警
10. **状态管理**：实时显示系统状态和检测结果

## 技术指标

- 垂直测量误差 < 2mm
- 水平测量误差 < 10mm
- 数据输出延迟 < 500ms
- 系统响应 < 200ms
- 检测准确率 > 99%

## 项目结构

```
├── config/           # 配置文件目录
│   └── config.yaml   # 主配置文件
├── install/          # 可执行文件目录
│   └── steel_wave_detection.exe  # 独立可执行文件（无需Python环境）
├── output/           # 检测结果输出目录
│   ├── detection_results.csv   # 检测结果CSV文件
│   ├── detection_results.json  # 检测结果JSON文件
├── src/              # 核心源代码
│   ├── utils/        # 工具函数
│   │   └── video_processor.py  # 视频处理工具
│   ├── __init__.py
│   ├── pipeline.py          # 实时处理管道（负责数据输出）
│   ├── wave_detector.py     # 浪形检测模块
│   └── web_server.py        # Web服务器模块
├── web/              # Web界面目录
│   └── templates/    # 模板文件
│       └── index.html       # 主页面
├── .gitignore        # Git忽略文件
├── README.md         # 项目说明
├── analyse_data_from_csv.py  # 数据分析工具脚本
├── debug.log         # 调试日志
├── main.py           # 桌面应用程序入口
├── main_adjust.py    # 调整后的桌面应用程序
├── requirements.txt  # 依赖列表
└── run_web.py        # Web服务器启动脚本
```

## 环境要求

- Python 3.8+
- OpenCV 4.8.0+
- NumPy 1.24.3+
- PyYAML 6.0.1+
- Flask 2.0.0+ (for Web server)
- PyQt5 5.15.0+ (for desktop app)
- matplotlib 3.7.2+

## 安装

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

## 配置

配置文件位于 `config/config.yaml`，包含以下主要部分：

- **camera**: 相机配置，包括类型、路径等
- **preprocessing**: 预处理配置，包括去雾、降噪等参数
- **detection**: 浪形检测配置，包括检测阈值等
- **visualization**: 可视化配置，包括显示参数等

## 运行系统

### 桌面应用程序（推荐）

```bash
# 运行桌面应用程序（推荐使用main_adjust.py）
python main_adjust.py

# 或者使用main.py
python main.py
```

运行桌面应用程序后，系统会显示主界面，提供以下功能：

- **常规模式**：仅展示原始监控画面和状态分析，适合日常监控使用
- **调试模式**：提供完整的功能，包括四路视频显示和详细日志记录
- **使用说明**：查看系统使用说明
- **退出系统**：关闭应用程序

### Web服务器

```bash
# 运行Web服务器
python run_web.py

# 访问地址: http://localhost:5001
```

### 独立可执行文件（无需Python环境）

```bash
# 直接运行可执行文件（Windows）
install/steel_wave_detection.exe
```

**注意**：`install/steel_wave_detection.exe` 是一个独立的可执行文件，无需安装Python环境即可运行。它包含了所有必要的依赖库，可以直接在Windows系统上运行。

### 数据分析工具

```bash
# 运行数据分析脚本
python analyse_data_from_csv.py
```

**功能说明**：
- 读取 `output/detection_results.csv` 文件中的检测数据
- 将Unix时间戳转换为北京时间(UTC+8)
- 统计浪形出现的帧数、类型分布、等级分布
- 计算浪高/浪宽的统计信息（平均值、最大值、最小值等）
- 生成分析报告到 `output/wave_analysis_report.txt`
- 导出浪形事件到 `output/wave_events.csv`

**输出文件**：
- `output/detection_results.csv`：检测系统运行时自动生成的检测结果
- `output/wave_events.csv`：筛选出的浪形事件记录
- `output/wave_analysis_report.txt`：详细的分析报告

## 数据输出

检测系统运行时会自动生成以下输出文件：

### 输出文件说明

| 文件 | 说明 | 生成位置 |
|------|------|----------|
| `detection_results.csv` | 完整的检测结果记录（时间戳、浪形类型、等级、浪高、浪宽等） | `output/` |
| `detection_results.json` | JSON格式的检测结果记录 | `output/` |

### 数据字段说明

CSV文件包含以下字段：
- `timestamp`：Unix时间戳（秒）
- `wave_type`：浪形类型（平直、双边浪、DS单边浪、WS单边浪等）
- `wave_level`：浪形等级（无、低、中、高、严重）
- `wave_height`：浪高（mm）
- `wave_width`：浪宽（mm）
- `wave_position_x`：浪形X坐标（mm）
- `wave_position_y`：浪形Y坐标（mm）

### 数据流向

```
视频输入 → 检测算法处理 → 检测结果 → CSV/JSON文件
                                          ↓
                              数据分析工具 (analyse_data_from_csv.py)
                                          ↓
                              分析报告 + 浪形事件CSV
```

## Web界面功能

1. **系统总览**：显示系统状态、视频文件列表和系统信息
2. **实时监控**：显示实时视频流、检测结果和性能指标
3. **参数配置**：调整算法参数，如去雾强度、透射率等
4. **历史日志**：查看历史检测记录
5. **报警记录**：查看系统报警信息

## 浪形类型

系统支持以下浪形类型：

- **平直**：带钢表面平直，无明显浪形
- **双边浪**：带钢两侧同时出现浪形
- **WS侧单边浪**：操作侧单边浪
- **DS侧单边浪**：驱动侧单边浪

## 浪形等级

系统将浪形分为以下等级：

- **无**：无浪形
- **低**：浪高 < 1.0mm
- **中**：1.0mm ≤ 浪高 < 2.0mm
- **高**：浪高 ≥ 2.0mm

## 性能优化

1. **多线程处理**：使用多线程并行处理帧图像，提高处理速度
2. **算法优化**：优化预处理和检测算法，降低计算复杂度
3. **内存管理**：合理管理内存，避免内存泄漏

## 异常处理

系统具有以下异常处理能力：

- **未检测到带钢**：当未检测到带钢时显示警告
- **处理错误**：当处理过程中出现错误时记录日志

## 扩展与定制

### 调整检测参数

修改 `config/config.yaml` 中的检测参数，包括浪形检测阈值等。

### 优化算法

在 `src/wave_detector.py` 中修改浪形检测算法，提高检测准确率。

## 测试与验证

### 功能测试

```bash
# 运行Web服务器进行功能测试
python run_web.py

# 运行桌面应用程序进行功能测试
python main_adjust.py
```

### 性能测试

检查日志文件 `debug.log` 中的性能指标。系统能够实时处理视频流，满足工业生产线上的实时检测需求。

## 项目维护

### 日志管理

定期清理 `debug.log` 文件，避免占用过多磁盘空间。

### 依赖更新

定期更新 `requirements.txt` 中的依赖包版本，确保系统稳定性和安全性。

## 注意事项

1. **环境要求**：确保系统环境满足Python 3.8+的要求
2. **视频文件**：确保提供的视频文件格式正确且可访问
3. **系统维护**：定期检查系统运行状态，确保正常运行

## 联系方式

- Author: ziyangsun12, GettingVia et al
- Email: ziyangsun@sjtu.edu.cn
- Project URL: https://github.com/ziyangsun12/steel-wave-detection

---

**© 2026 智能带钢浪形检测与分析系统**