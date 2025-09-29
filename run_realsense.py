#!/usr/bin/env python3
"""
RealSense YOLO-3D Main Script
Real-time 3D object detection using YOLOv11 and Intel RealSense stereo camera
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from detection_model import ObjectDetector
from realsense_camera import RealSenseCamera
from realsense_depth import RealSenseDepthEstimator
from realsense_bbox3d_utils import RealSenseBBox3DEstimator, RealSenseBirdEyeView

def main():
    """Main function for RealSense YOLO-3D."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Camera settings
    camera_width = 640
    camera_height = 480
    camera_fps = 30
    
    # Output settings
    output_path = "realsense_output.mp4"  # Path to output video file
    save_camera_params = True  # Save camera parameters to file
    
    # Model settings
    yolo_model_size = "nano"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    
    # Device settings
    device = 'cpu'  # Force CPU for stability
    
    # Detection settings
    conf_threshold = 0.25  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    
    # Feature toggles
    enable_tracking = True  # Enable object tracking
    enable_bev = True  # Enable Bird's Eye View visualization
    enable_3d_visualization = True  # Enable 3D visualization
    show_depth_map = True  # Show depth map visualization
    
    # Camera parameters file
    camera_params_file = "realsense_camera_params.json"
    # ===============================================
    
    print("=== RealSense YOLO-3D Object Detection ===")
    print(f"Using device: {device}")
    
    # Initialize RealSense camera
    print("Initializing RealSense camera...")
    camera = RealSenseCamera(
        width=camera_width,
        height=camera_height,
        fps=camera_fps,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False
    )
    
    if not camera.start():
        print("Error: Could not start RealSense camera")
        print("Make sure the camera is connected and not being used by another application")
        return
    
    # Save camera parameters if requested
    if save_camera_params:
        camera.save_camera_params(camera_params_file)
    
    # Initialize YOLO object detector
    print("Initializing YOLO object detector...")
    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu'
        )
    
    # Initialize RealSense depth estimator
    print("Initializing RealSense depth estimator...")
    depth_estimator = RealSenseDepthEstimator(camera)
    
    # Initialize 3D bounding box estimator
    print("Initializing 3D bounding box estimator...")
    bbox3d_estimator = RealSenseBBox3DEstimator(camera)
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        print("Initializing Bird's Eye View...")
        bev = RealSenseBirdEyeView(scale=60, size=(400, 400))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, camera_fps, (camera_width, camera_height))
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    print("Starting processing...")
    print("Press 'q' to quit, 's' to save screenshot, 'p' to pause/resume")
    
    paused = False
    
    # Main loop
    while True:
        # Check for key press at the beginning of each loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Exiting program...")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
            continue
        elif key == ord('s'):
            # Save screenshot
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, result_frame)
            print(f"Screenshot saved: {screenshot_path}")
        
        if paused:
            continue
            
        try:
            # Get frames from RealSense camera
            color_frame, depth_frame, _ = camera.get_frames()
            
            if color_frame is None:
                print("Warning: No color frame received")
                continue
            
            # Make copies for different visualizations
            original_frame = color_frame.copy()
            detection_frame = color_frame.copy()
            result_frame = color_frame.copy()
            
            # Step 1: Object Detection
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error in object detection: {e}")
                detections = []
            
            # Step 2: Get real depth data
            try:
                depth_map, _ = depth_estimator.estimate_depth(color_frame)
            except Exception as e:
                print(f"Error in depth estimation: {e}")
                depth_map = None
            
            # Step 3: Process 3D bounding boxes
            boxes_3d = []
            if detections and depth_map is not None:
                for detection in detections:
                    try:
                        # Extract detection information
                        if len(detection) >= 6:  # With tracking
                            x1, y1, x2, y2, conf, class_id, track_id = detection[:7]
                            object_id = int(track_id) if track_id is not None else None
                        else:  # Without tracking
                            x1, y1, x2, y2, conf, class_id = detection[:6]
                            object_id = None
                        
                        # Get class name
                        class_name = detector.class_names[int(class_id)]
                        
                        # Create 2D bounding box
                        bbox_2d = [int(x1), int(y1), int(x2), int(y2)]
                        
                        # Estimate 3D bounding box
                        box_3d = bbox3d_estimator.estimate_3d_box(bbox_2d, class_name, object_id)
                        
                        if box_3d['valid']:
                            boxes_3d.append(box_3d)
                            
                            # Draw 3D box on result frame
                            if enable_3d_visualization:
                                result_frame = bbox3d_estimator.draw_3d_box(result_frame, box_3d)
                        
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue
            
            # Step 4: Update Bird's Eye View
            if enable_bev and boxes_3d:
                try:
                    bev_image = bev.update(boxes_3d)
                except Exception as e:
                    print(f"Error updating BEV: {e}")
                    bev_image = np.zeros((400, 400, 3), dtype=np.uint8)
            else:
                bev_image = np.zeros((400, 400, 3), dtype=np.uint8)
            
            # Step 5: Create depth visualization
            if show_depth_map and depth_map is not None:
                try:
                    depth_colormap = depth_estimator.create_depth_colormap(depth_map)
                except Exception as e:
                    print(f"Error creating depth colormap: {e}")
                    depth_colormap = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            else:
                depth_colormap = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            
            # Step 6: Create combined visualization
            # Resize images to fit in display
            display_height = 300
            display_width = int(display_height * camera_width / camera_height)
            
            # Resize main frames
            result_display = cv2.resize(result_frame, (display_width, display_height))
            depth_display = cv2.resize(depth_colormap, (display_width, display_height))
            
            # Create combined view
            combined_view = np.zeros((display_height * 2, display_width + 400, 3), dtype=np.uint8)
            
            # Top row: main view and depth
            combined_view[:display_height, :display_width] = result_display
            combined_view[:display_height, display_width:] = cv2.resize(bev_image, (400, display_height))
            
            # Bottom row: depth map
            combined_view[display_height:, :display_width] = depth_display
            
            # Add labels
            cv2.putText(combined_view, "3D Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, "Bird's Eye View", (display_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, "Depth Map", (10, display_height + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS counter
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                fps_display = f"FPS: {fps:.1f}"
            
            cv2.putText(combined_view, fps_display, (10, combined_view.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add object count
            object_count = len(boxes_3d)
            cv2.putText(combined_view, f"Objects: {object_count}", (10, combined_view.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display combined view
            cv2.imshow('RealSense YOLO-3D', combined_view)
            
            # Write frame to output video
            out.write(result_frame)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue
    
    # Cleanup
    print("Cleaning up...")
    camera.stop()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to: {output_path}")
    print(f"Camera parameters saved to: {camera_params_file}")

if __name__ == "__main__":
    main()
