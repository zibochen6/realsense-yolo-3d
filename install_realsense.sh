#!/bin/bash
# RealSense YOLO-3D Installation Script

echo "=== RealSense YOLO-3D Installation ==="

# Check if running on Ubuntu/Debian
if command -v apt &> /dev/null; then
    echo "Detected Ubuntu/Debian system"
    
    # Install RealSense SDK
    echo "Installing Intel RealSense SDK..."
    
    # Add Intel RealSense repository
    sudo mkdir -p /etc/apt/keyrings
    curl -sSf https://librealsense.intel.com/Debian/apt-repo/conf/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/librealsense.list
    
    # Update package list and install
    sudo apt update
    sudo apt install -y librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg
    
    echo "✓ RealSense SDK installed"
    
elif command -v brew &> /dev/null; then
    echo "Detected macOS system"
    echo "Installing Intel RealSense SDK via Homebrew..."
    brew install librealsense
    echo "✓ RealSense SDK installed"
    
else
    echo "⚠️  Unsupported system. Please install Intel RealSense SDK manually."
    echo "Visit: https://github.com/IntelRealSense/librealsense/releases"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✓ Python dependencies installed"

# Test installation
echo "Testing installation..."
python test_realsense_setup.py

echo "Installation complete!"
echo "Run 'python run_realsense.py' to start the application"
