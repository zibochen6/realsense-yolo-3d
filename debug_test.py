#!/usr/bin/env python3
"""
Debug test to identify the exact error
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from detection_model import ObjectDetector
from depth_model import DepthEstimator

def debug_test():
    print("=== Debug Test ===")
    
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
            print("   ✓ RealSense camera working")
            pipeline.stop()
        else:
            print("   ✗ RealSense camera failed")
            return False
    except Exception as e:
        print(f"   ✗ RealSense camera error: {e}")
        return False
    
    # Test 2: Depth estimator
    print("2. Testing depth estimator...")
    try:
        depth_estimator = DepthEstimator(model_size='small', device='cpu')
        print("   ✓ Depth estimator loaded")
        
        # Test depth estimation
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = depth_estimator.estimate_depth(test_image)
        print(f"   ✓ Depth estimation result type: {type(result)}")
        print(f"   ✓ Depth estimation result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
    except Exception as e:
        print(f"   ✗ Depth estimator error: {e}")
        return False
    
    # Test 3: YOLO detector
    print("3. Testing YOLO detector...")
    try:
        detector = ObjectDetector(model_size="nano", device="cpu")
        print("   ✓ YOLO detector loaded")
    except Exception as e:
        print(f"   ✗ YOLO detector error: {e}")
        return False
    
    print("\n✓ All components working!")
    return True

if __name__ == "__main__":
    debug_test()
