#!/usr/bin/env python3
"""
Test RealSense camera with depth stream only
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_depth_only():
    print("Testing RealSense camera (depth stream only)...")
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure only depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
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
                depth_frame = frames.get_depth_frame()
                
                if depth_frame:
                    print(f"✓ Frame {i+1}: Depth {depth_frame.get_width()}x{depth_frame.get_height()}")
                    
                    # Convert to numpy array and display
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imshow('RealSense Depth', depth_colormap)
                    cv2.waitKey(1000)  # Display for 1 second
                else:
                    print(f"✗ Frame {i+1}: No depth frame received")
                    
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
    test_depth_only()
