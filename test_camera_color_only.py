#!/usr/bin/env python3
"""
Test RealSense camera with color stream only
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_color_only():
    print("Testing RealSense camera (color stream only)...")
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure only color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start pipeline
        print("Starting pipeline...")
        profile = pipeline.start(config)
        print("Pipeline started successfully!")
        
        # Get device info
        device = profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        
        # Try to get frames with longer timeout
        print("Waiting for frames (10 second timeout)...")
        for i in range(5):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    print(f"✓ Frame {i+1}: Color {color_frame.get_width()}x{color_frame.get_height()}")
                    
                    # Convert to numpy array and display
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.imshow('RealSense Color', color_image)
                    cv2.waitKey(1000)  # Display for 1 second
                else:
                    print(f"✗ Frame {i+1}: No color frame received")
                    
            except Exception as e:
                print(f"✗ Frame {i+1}: Error - {e}")
        
        # Stop pipeline
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Test completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    test_color_only()
