#!/usr/bin/env python3
"""
Simple RealSense camera test
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_camera():
    print("Testing RealSense camera...")
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        # Start pipeline
        print("Starting pipeline...")
        profile = pipeline.start(config)
        print("Pipeline started successfully!")
        
        # Get device info
        device = profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Try to get frames
        print("Waiting for frames...")
        for i in range(10):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    print(f"✓ Frame {i+1}: Color {color_frame.get_width()}x{color_frame.get_height()}, Depth {depth_frame.get_width()}x{depth_frame.get_height()}")
                else:
                    print(f"✗ Frame {i+1}: No frames received")
                    
            except Exception as e:
                print(f"✗ Frame {i+1}: Error - {e}")
        
        # Stop pipeline
        pipeline.stop()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    test_camera()
