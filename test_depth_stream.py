#!/usr/bin/env python3
"""
Test RealSense depth stream with different configurations
"""

import pyrealsense2 as rs
import numpy as np
import cv2

def test_depth_configurations():
    print("Testing different RealSense depth configurations...")
    
    # Try different configurations
    configs = [
        {"name": "D415 Default", "width": 640, "height": 480, "fps": 30},
        {"name": "D415 Low Res", "width": 424, "height": 240, "fps": 30},
        {"name": "D415 High FPS", "width": 640, "height": 480, "fps": 15},
        {"name": "D415 Very Low", "width": 320, "height": 240, "fps": 15},
    ]
    
    for config_info in configs:
        print(f"\n--- Testing {config_info['name']} ---")
        
        try:
            # Create pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            config.enable_stream(rs.stream.color, config_info['width'], config_info['height'], rs.format.bgr8, config_info['fps'])
            config.enable_stream(rs.stream.depth, config_info['width'], config_info['height'], rs.format.z16, config_info['fps'])
            
            # Start pipeline
            profile = pipeline.start(config)
            
            # Get device info
            device = profile.get_device()
            print(f"Device: {device.get_info(rs.camera_info.name)}")
            
            # Try to get frames
            print("Waiting for frames...")
            for i in range(5):
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if color_frame and depth_frame:
                        print(f"✓ Frame {i+1}: Color {color_frame.get_width()}x{color_frame.get_height()}, Depth {depth_frame.get_width()}x{depth_frame.get_height()}")
                        
                        # Test depth data
                        depth_data = np.asanyarray(depth_frame.get_data())
                        valid_pixels = np.sum(depth_data > 0)
                        total_pixels = depth_data.size
                        print(f"  Valid depth pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
                        
                        if i == 0:  # Show first frame
                            # Display frames
                            color_image = np.asanyarray(color_frame.get_data())
                            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)
                            
                            cv2.imshow('Color', color_image)
                            cv2.imshow('Depth', depth_colormap)
                            cv2.waitKey(1000)
                            
                    else:
                        print(f"✗ Frame {i+1}: Missing frames")
                        
                except Exception as e:
                    print(f"✗ Frame {i+1}: Error - {e}")
            
            # Stop pipeline
            pipeline.stop()
            print(f"✓ {config_info['name']} configuration works!")
            return config_info
            
        except Exception as e:
            print(f"✗ {config_info['name']} failed: {e}")
            try:
                pipeline.stop()
            except:
                pass
            continue
    
    print("\n✗ No working depth configuration found")
    return None

if __name__ == "__main__":
    working_config = test_depth_configurations()
    cv2.destroyAllWindows()
    
    if working_config:
        print(f"\n🎉 Working configuration found: {working_config['name']}")
        print(f"Use these settings: width={working_config['width']}, height={working_config['height']}, fps={working_config['fps']}")
    else:
        print("\n⚠️ No working depth configuration found. You may need to:")
        print("1. Check camera firmware version")
        print("2. Try different USB port")
        print("3. Check camera permissions")
        print("4. Restart the camera")
