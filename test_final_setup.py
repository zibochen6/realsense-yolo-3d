#!/usr/bin/env python3
"""
Final test to verify the working RealSense setup
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from detection_model import ObjectDetector
from depth_model import DepthEstimator

def test_final_setup():
    print("=== Final RealSense Setup Test ===")
    
    # Test 1: RealSense camera
    print("1. Testing RealSense camera...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = np.asanyarray(frames.get_color_frame().get_data())
        
        if color_frame is not None:
            print("   ✓ RealSense camera working - Color stream active")
            pipeline.stop()
        else:
            print("   ✗ RealSense camera failed")
            return False
    except Exception as e:
        print(f"   ✗ RealSense camera error: {e}")
        return False
    
    # Test 2: YOLO detector
    print("2. Testing YOLO detector...")
    try:
        detector = ObjectDetector(model_size="nano", device="cpu")
        print("   ✓ YOLO detector loaded successfully")
    except Exception as e:
        print(f"   ✗ YOLO detector error: {e}")
        return False
    
    # Test 3: Depth estimator
    print("3. Testing depth estimator...")
    try:
        depth_estimator = DepthEstimator(model_size="small", device="cpu")
        print("   ✓ Depth estimator loaded successfully")
    except Exception as e:
        print(f"   ✗ Depth estimator error: {e}")
        return False
    
    print("\n�� All tests passed! Your RealSense YOLO-3D setup is working!")
    print("\nTo run the application:")
    print("python run_realsense_working.py")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'p' to pause/resume")
    print("- Press 's' to save screenshot")
    
    return True

if __name__ == "__main__":
    test_final_setup()
