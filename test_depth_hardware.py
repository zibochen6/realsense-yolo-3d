#!/usr/bin/env python3
"""
Test RealSense depth hardware and settings
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_depth_hardware():
    print("Testing RealSense depth hardware...")
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable only depth stream first
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Get depth sensor
        depth_sensor = device.first_depth_sensor()
        print(f"Depth sensor: {depth_sensor.get_info(rs.camera_info.name)}")
        
        # Check depth sensor options
        print("\nDepth sensor options:")
        for option in depth_sensor.get_supported_options():
            try:
                value = depth_sensor.get_option(option)
                print(f"  {option}: {value}")
            except:
                pass
        
        # Try to get frames with longer timeout
        print("\nTrying to get depth frames...")
        for i in range(3):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                depth_frame = frames.get_depth_frame()
                
                if depth_frame:
                    print(f"✓ Got depth frame {i+1}: {depth_frame.get_width()}x{depth_frame.get_height()}")
                    depth_data = np.asanyarray(depth_frame.get_data())
                    print(f"  Depth range: {np.min(depth_data)} - {np.max(depth_data)}")
                    print(f"  Valid pixels: {np.sum(depth_data > 0)}/{depth_data.size}")
                else:
                    print(f"✗ No depth frame {i+1}")
                    
            except Exception as e:
                print(f"✗ Error getting frame {i+1}: {e}")
        
        pipeline.stop()
        
    except Exception as e:
        print(f"Error: {e}")
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    test_depth_hardware()
