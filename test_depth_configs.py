#!/usr/bin/env python3
"""
Test different RealSense depth stream configurations
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_depth_configs():
    print("=== Testing Different Depth Configurations ===")
    
    # Test different configurations
    configs = [
        {"width": 640, "height": 480, "fps": 30, "format": rs.format.z16},
        {"width": 640, "height": 480, "fps": 15, "format": rs.format.z16},
        {"width": 640, "height": 480, "fps": 6, "format": rs.format.z16},
        {"width": 424, "height": 240, "fps": 30, "format": rs.format.z16},
        {"width": 424, "height": 240, "fps": 15, "format": rs.format.z16},
        {"width": 320, "height": 240, "fps": 30, "format": rs.format.z16},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Testing Config {i+1}: {config['width']}x{config['height']} @ {config['fps']}fps ---")
        
        pipeline = rs.pipeline()
        rs_config = rs.config()
        
        try:
            # Enable streams
            rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            rs_config.enable_stream(rs.stream.depth, 
                                  config['width'], 
                                  config['height'], 
                                  config['format'], 
                                  config['fps'])
            
            # Start pipeline
            profile = pipeline.start(rs_config)
            
            # Test frame capture with shorter timeout
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            print(f"  Color frame: {color_frame is not None}")
            print(f"  Depth frame: {depth_frame is not None}")
            
            if depth_frame:
                depth_data = np.asanyarray(depth_frame.get_data())
                print(f"  ✓ Depth working! Shape: {depth_data.shape}")
                pipeline.stop()
                return config  # Return working config
            else:
                print(f"  ✗ Depth frame is None")
            
            pipeline.stop()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            try:
                pipeline.stop()
            except:
                pass
    
    print("\n✗ No working depth configuration found")
    return None

def test_color_only():
    print("\n=== Testing Color Stream Only ===")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # Enable only color stream
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = pipeline.start(config)
        
        # Test frame capture
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        color_frame = frames.get_color_frame()
        
        print(f"Color frame: {color_frame is not None}")
        
        if color_frame:
            color_data = np.asanyarray(color_frame.get_data())
            print(f"✓ Color working! Shape: {color_data.shape}")
            pipeline.stop()
            return True
        else:
            print(f"✗ Color frame is None")
        
        pipeline.stop()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        try:
            pipeline.stop()
        except:
            pass
    
    return False

if __name__ == "__main__":
    # Test color stream first
    color_works = test_color_only()
    
    # Test depth configurations
    working_config = test_depth_configs()
    
    if working_config:
        print(f"\n✓ Found working depth config: {working_config}")
    else:
        print(f"\n✗ No working depth config found")
        if color_works:
            print("✓ Color stream works, can use depth estimation fallback")
        else:
            print("✗ Neither color nor depth streams work")
