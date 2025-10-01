# YOLO-3D with Intel RealSense

A real-time 3D object detection system that combines YOLOv11 for object detection with Intel RealSense 3D cameras to create accurate 3D bounding boxes with real depth data, segmentation overlays, and unique object coloring.

## Features

- **Real-time object detection** using YOLOv11
- **Real depth data** from Intel RealSense cameras (D415, D435i, etc.)
- **3D bounding box visualization** as proper 3D cubes
- **Object segmentation** with semi-transparent overlays
- **Unique colors** for each tracked object
- **Object tracking** capabilities with Kalman filtering
- **Fallback support** to Depth Anything v2 if RealSense depth is unavailable
- **Multiple camera support** with automatic configuration
- **Real-time performance** with optimized processing

## Hardware Requirements

- **Intel RealSense Camera**: D415, D435i, D455, or compatible model
- **USB 3.0 port** for camera connection
- **Minimum 4GB RAM** (8GB recommended)
- **CPU**: Any modern processor (GPU acceleration optional)

## Software Requirements

- Python 3.8+
- PyTorch 2.0+
- Intel RealSense SDK 2.0
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/niconielsen32/YOLO-3D.git
cd YOLO-3D
```

### 2. Install Intel RealSense SDK

#### Option A: Automated Installation (Recommended)
```bash
chmod +x install_realsense.sh
./install_realsense.sh
```

#### Option B: Manual Installation
```bash
# Add Intel's repository
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

# Install RealSense SDK
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg

# Install Python bindings
pip install pyrealsense2
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python test_realsense_setup.py
```

## Usage

### Quick Start
```bash
python run_realsense_3d_segmentation.py
```

### Available Scripts

| Script | Description |
|--------|-------------|
| `run_realsense_3d_segmentation.py` | **Main application** - 3D detection with segmentation and unique colors |
| `test_realsense_setup.py` | Verify camera and dependencies |
| `realsense_calibration.py` | Camera calibration and depth accuracy testing (optional) |

### Configuration Options

You can modify the following parameters in the main script:

```python
# Camera settings
camera_width = 640
camera_height = 480
camera_fps = 6

# Model settings
detector = ObjectDetector(model_size="nano", device="cpu")  # or "cuda" for GPU
```

### Controls

- **'q' or ESC**: Quit the application
- **Real-time display**: Shows 3D detection on left, depth map on right

## Project Structure

```
YOLO-3D/
├── run_realsense_3d_segmentation.py  # Main application
├── realsense_camera.py               # RealSense camera interface
├── realsense_depth.py                # Real depth processing
├── realsense_bbox3d_utils.py         # 3D bounding box utilities
├── detection_model.py                # YOLOv11 object detection
├── test_realsense_setup.py           # Setup verification
├── realsense_calibration.py          # Camera calibration (optional)
├── install_realsense.sh              # Automated installation
├── requirements.txt                  # Python dependencies
├── yolo11n.pt                        # YOLO model weights
└── README.md                         # This file
```

## How It Works

1. **Camera Initialization**: RealSense camera starts with color and depth streams
2. **Object Detection**: YOLOv11 detects objects and provides 2D bounding boxes
3. **Real Depth Processing**: RealSense provides accurate depth data for each pixel
4. **3D Box Creation**: Combines 2D boxes with real depth to create 3D cubes
5. **Segmentation**: Applies semi-transparent overlays to detected objects
6. **Unique Coloring**: Assigns distinct colors to each tracked object
7. **Visualization**: Displays 3D cubes with depth information and segmentation

## Key Features Explained

### Real Depth vs Estimated Depth
- **Real Depth**: Uses actual distance measurements from RealSense stereo cameras
- **Estimated Depth**: Falls back to AI-based depth estimation if RealSense depth fails
- **Accuracy**: Real depth is significantly more accurate for distance measurements

### 3D Bounding Boxes
- **Proper 3D Cubes**: Drawn as actual 3D wireframe cubes, not flat rectangles
- **Depth-based Sizing**: Cube depth varies based on actual object distance
- **Perspective**: Includes connecting lines and shading for 3D effect

### Segmentation Overlays
- **Semi-transparent**: Objects are highlighted with colored overlays
- **Class-based**: Different colors for different object types
- **Track-based**: Unique colors for each tracked object instance

### Object Tracking
- **Kalman Filtering**: Smooth tracking across frames
- **Unique IDs**: Each object gets a persistent ID
- **Color Consistency**: Same object maintains same color throughout tracking

## GPU Setup for Jetson Devices

### Current Status
The application currently runs with CPU-only PyTorch. For GPU acceleration on Jetson devices, you need CUDA-enabled PyTorch.

### Why GPU Setup is Challenging
- Jetson devices require special NVIDIA-built PyTorch wheels (not standard PyTorch)
- Official PyTorch wheels for JetPack 6.0 have broken download links
- Standard PyTorch wheels are built for x86_64, not ARM64 Jetson devices

### GPU Setup Options

#### Option 1: Upgrade JetPack (Recommended)
Upgrade to JetPack 6.1 or 6.2 where PyTorch wheels are more readily available:
```bash
# Check current version
cat /etc/nv_tegra_release

# Upgrade to JetPack 6.1 (if available)
sudo apt update
sudo apt upgrade
```

#### Option 2: Manual PyTorch Compilation
Build PyTorch from source with CUDA support:
```bash
# Install dependencies
sudo apt-get install build-essential cmake git

# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Set environment variables
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.7"  # For Jetson AGX Orin

# Build and install
python setup.py install
```

#### Option 3: Use Jetson Containers
Use pre-built containers with GPU support:
```bash
# Install jetson-containers
git clone https://github.com/dusty-nv/jetson-containers.git
cd jetson-containers
bash install.sh

# Run with PyTorch container
jetson-containers run dustynv/pytorch:2.5
```

### Performance Impact
- **CPU mode**: ~5-10 FPS (functional but slower)
- **GPU mode**: ~20-30 FPS (significantly faster)

### Verification
To check if GPU is working:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Troubleshooting

### Camera Not Detected
```bash
# Check USB connection
lsusb | grep Intel

# Test with RealSense viewer
realsense-viewer
```

### Depth Stream Issues
- Try different FPS settings (6fps works well for most cameras)
- Check camera firmware version
- Ensure good lighting conditions
- Verify USB 3.0 connection

### Performance Issues
- Use smaller model sizes ("nano" instead of "large")
- Reduce camera resolution
- Use CPU instead of GPU if memory limited

### Installation Issues
```bash
# Reinstall RealSense SDK
sudo apt-get remove librealsense2-*
./install_realsense.sh

# Reinstall Python dependencies
pip install --upgrade -r requirements.txt
```

## Performance Tips

1. **Model Size**: Use "nano" for best performance, "large" for best accuracy
2. **Resolution**: Lower resolution (640x480) for better FPS
3. **FPS**: 6fps is optimal for most RealSense cameras
4. **Device**: Use GPU if available for faster processing

## Supported Cameras

- **Intel RealSense D415**: Good for indoor use, reliable depth
- **Intel RealSense D435i**: Includes IMU, excellent for robotics
- **Intel RealSense D455**: Longer range, better outdoor performance
- **Intel RealSense L515**: LiDAR-based, high accuracy

## Future Enhancements

- [ ] Support for multiple cameras
- [ ] Point cloud visualization
- [ ] 3D object reconstruction
- [ ] Integration with ROS
- [ ] Mobile app interface
- [ ] Cloud processing support

## Acknowledgments

- **YOLOv11** by Ultralytics
- **Intel RealSense SDK** by Intel Corporation
- **Depth Anything v2** by Microsoft (fallback support)
- **OpenCV** for computer vision utilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with RealSense camera
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the setup verification script
- Open an issue on GitHub with camera model and error details
