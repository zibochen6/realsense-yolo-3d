#!/usr/bin/env python3
"""
Test RealSense depth stream in detail to understand the issue
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_depth_stream():
    print("=== Testing RealSense Depth Stream ===")
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get device info
        device = profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Get depth sensor
        depth_sensor = device.first_depth_sensor()
        print(f"Depth sensor: {depth_sensor}")
        
        # Get depth scale
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale}")
        
        # Get depth options
        print("\nDepth sensor options:")
        for option in depth_sensor.get_supported_options():
            try:
                value = depth_sensor.get_option(option)
                print(f"  {option}: {value}")
            except:
                print(f"  {option}: <error getting value>")
        
        # Test frame capture
        print("\nTesting frame capture...")
        for i in range(5):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            print(f"Frame {i+1}:")
            print(f"  Color frame: {color_frame is not None}")
            print(f"  Depth frame: {depth_frame is not None}")
            
            if depth_frame:
                depth_data = np.asanyarray(depth_frame.get_data())
                print(f"  Depth data shape: {depth_data.shape}")
                print(f"  Depth data type: {depth_data.dtype}")
                print(f"  Depth range: {depth_data.min()} - {depth_data.max()}")
                
                # Convert to meters
                depth_meters = depth_data.astype(np.float32) * depth_scale
                valid_depths = depth_meters[depth_meters > 0]
                if len(valid_depths) > 0:
                    print(f"  Valid depth range: {valid_depths.min():.3f} - {valid_depths.max():.3f} meters")
                else:
                    print(f"  No valid depth values found")
        
        pipeline.stop()
        print("\n✓ Depth stream test completed successfully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    test_depth_stream()
