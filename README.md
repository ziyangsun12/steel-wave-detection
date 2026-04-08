# Hot Rolling Steel Strip Wave Detection System

## Project Overview

The Hot Rolling Steel Strip Wave Detection System is a real-time detection system based on Python and computer vision, designed to detect wave defects in steel strips between finishing stands in hot rolling mills. The system can recover the 3D dynamic contour of the steel strip surface from industrial camera video streams, automatically identify and classify wave types, and provide data support for real-time control between stands.

## Core Features

1. **Video Stream Access**: Supports industrial cameras, local videos, and folder image reading
2. **High-temperature Water Mist Preprocessing**: Defogging, denoising, contrast enhancement, steel strip region segmentation
3. **3D Contour Reconstruction**: Plane fitting, height difference calculation, centerline/edge extraction
4. **Wave Detection Classifier**: Based on contour features + YOLOv8 classification, output type + level
5. **Quantitative Measurement**: Output wave height, width, position (mm)
6. **Real-time Pipeline**: Multi-threading/async processing, meeting millisecond-level indicators
7. **Visualization**: Real-time contour drawing, wave annotation, measurement value display
8. **Logging + Alarm + Data Reporting**: CSV/JSON result storage, PLC integration support
9. **Configuration**: Camera parameters, calibration data, thresholds, IO paths fully configurable
10. **Exception Handling**: Automatic fault tolerance for disconnection, overexposure, black screen, no steel strip

## Technical Indicators

- Vertical measurement error < 2mm
- Horizontal measurement error < 10mm
- Data output delay < 500ms
- System response < 200ms
- Failure rate > 99%

## Project Structure

```
├── config/           # Configuration files
│   └── config.yaml   # Main configuration file
├── data/             # Data directory
│   ├── sample_video.mp4  # Sample video
│   └── images/       # Sample images
├── logs/             # Log directory
├── output/           # Output directory
├── src/              # Core source code
│   ├── __init__.py
│   ├── preprocessing.py     # Preprocessing module
│   ├── 3d_reconstruction.py # 3D reconstruction module
│   ├── wave_detector.py     # Wave detection module
│   ├── visualization.py     # Visualization module
│   ├── pipeline.py          # Real-time processing pipeline
│   └── utils.py             # Utility functions
├── tests/            # Test directory
├── requirements.txt  # Dependency list
├── run.py            # Start script
└── README.md         # Project description
```

## Environment Requirements

- Python 3.8+
- OpenCV 4.8.0+
- NumPy 1.24.3+
- PyYAML 6.0.1+
- Ultralytics 8.0.196+ (for YOLOv8)
- scikit-learn 1.3.0+
- pandas 2.0.3+
- matplotlib 3.7.2+

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The configuration file is located at `config/config.yaml` and includes the following main sections:

- **camera**: Camera configuration, including type, path, resolution, etc.
- **calibration**: Calibration parameters, including pixel-to-millimeter conversion factors
- **preprocessing**: Preprocessing configuration, including defogging, denoising, contrast enhancement, etc.
- **reconstruction**: 3D reconstruction configuration, including reconstruction method, plane fitting, etc.
- **detection**: Wave detection configuration, including wave types, level thresholds, etc.
- **realtime**: Real-time processing configuration, including multi-threading, async processing, etc.
- **visualization**: Visualization configuration, including display parameters, plotting parameters, etc.
- **logging**: Log configuration, including level, file path, etc.
- **output**: Output configuration, including CSV, JSON output, etc.
- **exception**: Exception handling configuration, including disconnection, overexposure, black screen, etc.

## Running the System

### Basic Usage

```bash
# Run with default configuration
python run.py

# Use specified configuration file
python run.py --config config/config.yaml

# Process local video
python run.py --video data/sample_video.mp4

# Use industrial camera
python run.py --camera 0

# Process image folder
python run.py --folder data/images

# Enable debug mode
python run.py --debug
```

### Keyboard Controls

- `q`: Exit the system
- `s`: Save current frame

## Output Description

- **CSV Output**: `output/detection_results.csv`, includes timestamp, wave type, level, height, width, position, etc.
- **JSON Output**: `output/detection_results.json`, includes detailed detection results and processing information
- **Log Output**: `logs/detection.log`, includes system running logs
- **Image Output**: `output/capture.jpg`, saved screenshot

## Wave Types

The system supports the following wave types:

- **Flat**: Steel strip surface is flat, no obvious waves
- **DS Single Side Wave**: Drive side single side wave
- **WS Single Side Wave**: Work side single side wave
- **Double Side Wave**: Waves on both sides simultaneously
- **Center Wave**: Wave in the middle of the steel strip
- **Composite Wave**: Multiple wave types simultaneously

## Wave Levels

The system classifies waves into the following levels:

- **Low**: Wave height < 1.0mm
- **Medium**: 1.0mm ≤ Wave height < 2.0mm
- **High**: 2.0mm ≤ Wave height < 3.0mm
- **Severe**: Wave height ≥ 3.0mm

## Performance Optimization

1. **Multi-threading**: Use multi-threading to process frame images in parallel, improving processing speed
2. **Queue Buffering**: Use queue buffering for frame data to avoid processing delays
3. **Frame Dropping**: When the queue is full, drop old frames to ensure real-time performance
4. **Algorithm Optimization**: Optimize preprocessing and detection algorithms to reduce computational complexity
5. **Memory Management**: Manage memory properly to avoid memory leaks

## Exception Handling

The system has the following exception handling capabilities:

- **Disconnection Handling**: Automatic reconnection when camera disconnection occurs
- **Overexposure Handling**: Warning when overexposure is detected
- **Black Screen Handling**: Warning when black screen is detected
- **No Steel Strip Handling**: Warning when no steel strip is detected

## Extension and Customization

### Adding New Wave Types

Modify the `detection.wave_types` configuration in `config/config.yaml`, then implement the corresponding classification logic in `src/wave_detector.py`.

### Adjusting Detection Parameters

Modify the `detection.parameters` configuration in `config/config.yaml`, including minimum wave length, minimum wave height, and other parameters.

### Integrating New Cameras

Modify the `camera` configuration in `config/config.yaml` to add parameters for new cameras.

## Testing and Validation

### Unit Tests

```bash
cd tests
python -m pytest test_wave_detection.py -v
```

### Performance Testing

```bash
python run.py --video data/sample_video.mp4
```

Check the performance indicators in the log file `logs/detection.log`.

## Project Maintenance

### Log Management

Regularly clean up log files in the `logs/` directory to avoid excessive disk space usage.

### Dependency Updates

Regularly update the dependency package versions in `requirements.txt` to ensure system stability and security.

### Model Updates

If using YOLOv8 models, regularly update model files to improve detection accuracy.

## Notes

1. **Industrial Environment**: In high-temperature, high-water mist environments, ensure adequate camera protection measures
2. **Calibration**: Regularly calibrate camera parameters to ensure measurement accuracy
3. **System Maintenance**: Regularly check system operation status to ensure trouble-free operation
4. **Data Backup**: Regularly back up detection results and log files

## Contact

- Author: Builder
- Email: builder@example.com
- Project URL: https://github.com/builder/steel-wave-detection

---

**© 2026 Hot Rolling Steel Strip Wave Detection System**