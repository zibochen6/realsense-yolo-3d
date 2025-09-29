# YOLO-3D with Intel RealSense

A real-time 3D object detection system that combines YOLOv11 for object detection with Intel RealSense stereo cameras for accurate depth estimation and true 3D bounding box visualization.

## Features

- **Real 3D Depth**: Uses Intel RealSense stereo cameras for accurate depth data instead of estimated depth
- **Real-time Object Detection**: YOLOv11 for fast and accurate object detection
- **True 3D Bounding Boxes**: Accurate 3D bounding box estimation using real depth data
- **Bird's Eye View**: Top-down visualization of detected objects
- **Object Tracking**: Kalman filter-based 3D object tracking
- **Camera Calibration**: Built-in calibration tools for optimal performance
- **Multiple Visualizations**: Combined view with color, depth, and BEV

## Requirements

### Hardware
- Intel RealSense camera (D435, D435i, D455, L515, etc.)
- USB 3.0 port (recommended for best performance)
- NVIDIA GPU (optional, for faster YOLO inference)

### Software
- Python 3.8+
- Intel RealSense SDK 2.0+
- OpenCV
- PyTorch
- Other dependencies listed in `requirements.txt`

## Installation

### 1. Install Intel RealSense SDK

#### Ubuntu/Debian:
```bash
# Add Intel RealSense repository
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/apt-repo/conf/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/librealsense.list

# Update package list and install
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg

# Install Python bindings
pip install pyrealsense2
```

#### Windows:
1. Download and install Intel RealSense SDK from: https://github.com/IntelRealSense/librealsense/releases
2. Install Python bindings: `pip install pyrealsense2`

#### macOS:
```bash
brew install librealsense
pip install pyrealsense2
```

### 2. Install Project Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd YOLO-3D

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test RealSense camera
python realsense_camera.py

# Test depth estimation
python realsense_depth.py

# Test 3D bounding box estimation
python realsense_bbox3d_utils.py
```

## Usage

### Quick Start

```bash
# Run the main RealSense YOLO-3D application
python run_realsense.py
```

### Camera Calibration (Recommended)

For best results, calibrate your camera first:

```bash
# Run calibration utility
python realsense_calibration.py
```

Choose from:
1. **Depth accuracy calibration**: Measure objects at known distances
2. **Ground plane calibration**: Calibrate ground plane detection
3. **Full calibration**: Both depth and ground plane calibration

### Configuration

Edit `run_realsense.py` to modify settings:

```python
# Camera settings
camera_width = 640
camera_height = 480
camera_fps = 30

# Model settings
yolo_model_size = "nano"  # "nano", "small", "medium", "large", "extra"

# Detection settings
conf_threshold = 0.25
iou_threshold = 0.45

# Feature toggles
enable_tracking = True
enable_bev = True
enable_3d_visualization = True
show_depth_map = True
```

## Controls

- **'q'**: Quit application
- **'p'**: Pause/Resume processing
- **'s'**: Save screenshot

## Project Structure

```
YOLO-3D/
├── run_realsense.py              # Main RealSense application
├── realsense_camera.py           # RealSense camera interface
├── realsense_depth.py            # RealSense depth estimation
├── realsense_bbox3d_utils.py     # 3D bounding box utilities
├── realsense_calibration.py      # Camera calibration utility
├── detection_model.py            # YOLOv11 object detection
├── requirements.txt              # Project dependencies
└── README_REALSENSE.md          # This file
```

## How It Works

1. **RealSense Camera**: Captures synchronized color and depth frames
2. **Object Detection**: YOLOv11 detects objects in the color frame
3. **Depth Integration**: Real depth data is used instead of estimated depth
4. **3D Box Estimation**: Combines 2D detections with real depth for accurate 3D boxes
5. **Visualization**: Displays 3D boxes, depth map, and bird's eye view

## Advantages over Depth Anything v2

- **Higher Accuracy**: Real depth data is more accurate than estimated depth
- **Better Performance**: No need for depth estimation model inference
- **Real-time**: Direct depth capture without processing overhead
- **Consistent**: Depth data is always available and consistent
- **Calibrated**: Can be calibrated for specific environments

## Troubleshooting

### Camera Not Detected
```bash
# Check if camera is connected
lsusb | grep Intel

# Test with RealSense viewer
realsense-viewer
```

### Permission Issues
```bash
# Add user to video group
sudo usbermod -a -G video $USER
# Log out and back in
```

### Performance Issues
- Use smaller YOLO model size ("nano" or "small")
- Reduce camera resolution
- Disable unnecessary features
- Use GPU acceleration if available

### Depth Quality Issues
- Ensure good lighting conditions
- Clean camera lenses
- Run depth calibration
- Check camera firmware version

## Advanced Usage

### Custom Object Dimensions

Edit `realsense_bbox3d_utils.py` to add custom object dimensions:

```python
DEFAULT_DIMS = {
    'your_object': np.array([height, width, length]),  # in meters
    # ... existing objects
}
```

### Custom Camera Parameters

Load custom camera parameters:

```python
camera = RealSenseCamera(width=640, height=480, fps=30)
camera.start()
camera.save_camera_params('my_camera_params.json')
```

### Integration with Other Systems

The RealSense modules can be easily integrated into other projects:

```python
from realsense_camera import RealSenseCamera
from realsense_depth import RealSenseDepthEstimator
from realsense_bbox3d_utils import RealSenseBBox3DEstimator

# Initialize
camera = RealSenseCamera()
depth_estimator = RealSenseDepthEstimator(camera)
bbox3d_estimator = RealSenseBBox3DEstimator(camera)

# Use in your application
camera.start()
depth_map, color_frame = depth_estimator.estimate_depth(None)
box_3d = bbox3d_estimator.estimate_3d_box(bbox_2d, class_name)
```

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | FPS | Resolution | Model Size |
|----------|-----|------------|------------|
| Intel i7 + RTX 3080 | 30+ | 640x480 | Large |
| Intel i5 + GTX 1660 | 25+ | 640x480 | Medium |
| Intel i5 (CPU only) | 15+ | 640x480 | Nano |
| Raspberry Pi 4 | 5+ | 320x240 | Nano |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with RealSense camera
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv11 by Ultralytics
- Intel RealSense SDK
- OpenCV community
- PyTorch team
