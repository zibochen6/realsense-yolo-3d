#!/usr/bin/env python3
"""
Simple RealSense YOLO-3D - Working Version
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs

# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView

def main():
    print("=== Simple RealSense YOLO-3D ===")
    
    # Configuration
    camera_width = 640
    camera_height = 480
    camera_fps = 30
    
    # Initialize RealSense camera (color only)
    print("Initializing RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, camera_width, camera_height, rs.format.bgr8, camera_fps)
    
    try:
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        print("✓ RealSense camera started")
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return
    
    # Initialize models
    print("Initializing models...")
    try:
        detector = ObjectDetector(model_size="nano", device="cpu")
        depth_estimator = DepthEstimator(model_size="small", device="cpu")
        
        # Create camera parameters
        camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        projection_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx, 0],
            [0, color_intrinsics.fy, color_intrinsics.ppy, 0],
            [0, 0, 1, 0]
        ])
        
        bbox3d_estimator = BBox3DEstimator(
            camera_matrix=camera_matrix,
            projection_matrix=projection_matrix
        )
        
        bev = BirdEyeView(scale=60, size=(400, 400))
        
        print("✓ All models initialized")
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        pipeline.stop()
        return
    
    print("Starting processing...")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            # Get frames
            frames = pipeline.wait_for_frames()
            color_frame = np.asanyarray(frames.get_color_frame().get_data())
            
            if color_frame is None:
                continue
            
            # Object detection
            detection_frame, detections = detector.detect(color_frame.copy(), track=True)
            
            # Depth estimation
            depth_map = depth_estimator.estimate_depth(color_frame)
            
            # Process 3D boxes
            boxes_3d = []
            if detections and depth_map is not None:
                for detection in detections:
                    try:
                        if len(detection) >= 6:
                            x1, y1, x2, y2, conf, class_id, track_id = detection[:7]
                            object_id = int(track_id) if track_id is not None else None
                        else:
                            x1, y1, x2, y2, conf, class_id = detection[:6]
                            object_id = None
                        
                        class_name = detector.class_names[int(class_id)]
                        bbox_2d = [int(x1), int(y1), int(x2), int(y2)]
                        
                        # Get depth at center
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        depth_value = depth_map[int(center_y), int(center_x)]
                        
                        # Estimate 3D box
                        box_3d = bbox3d_estimator.estimate_3d_box(bbox_2d, depth_value, class_name, object_id)
                        
                        if box_3d['valid']:
                            boxes_3d.append(box_3d)
                            detection_frame = bbox3d_estimator.draw_3d_box(detection_frame, box_3d)
                        
                    except Exception as e:
                        print(f"Detection error: {e}")
                        continue
            
            # Update BEV
            bev_image = bev.update(boxes_3d)
            
            # Create depth visualization
            if depth_map is not None:
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            else:
                depth_colormap = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            
            # Create combined view
            display_height = 300
            display_width = int(display_height * camera_width / camera_height)
            
            result_display = cv2.resize(detection_frame, (display_width, display_height))
            depth_display = cv2.resize(depth_colormap, (display_width, display_height))
            
            combined_view = np.zeros((display_height * 2, display_width + 400, 3), dtype=np.uint8)
            combined_view[:display_height, :display_width] = result_display
            combined_view[:display_height, display_width:] = cv2.resize(bev_image, (400, display_height))
            combined_view[display_height:, :display_width] = depth_display
            
            # Add labels
            cv2.putText(combined_view, "3D Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, "Bird's Eye View", (display_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, "Depth Map", (10, display_height + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv2.putText(combined_view, f"FPS: {fps:.1f}", (10, combined_view.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Object count
            cv2.putText(combined_view, f"Objects: {len(boxes_3d)}", (10, combined_view.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Simple RealSense YOLO-3D', combined_view)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()
