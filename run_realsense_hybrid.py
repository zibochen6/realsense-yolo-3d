#!/usr/bin/env python3
"""
RealSense YOLO-3D Hybrid Version
Uses working color stream + fallback depth estimation
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
from depth_model import DepthEstimator  # Use original depth estimation as fallback

class HybridRealSenseCamera:
    """
    Hybrid RealSense camera that uses color stream + depth estimation fallback
    """
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize RealSense for color only
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure only color stream (we know this works)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # Initialize depth estimator as fallback
        self.depth_estimator = DepthEstimator(model_size='small', device='cpu')
        
        # Camera parameters
        self.camera_params = None
        
    def start(self):
        """Start the camera pipeline"""
        try:
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Create camera parameters
            self._create_camera_params()
            
            print(f"Hybrid RealSense camera started successfully")
            print(f"Color intrinsics: {self.color_intrinsics}")
            print("Using depth estimation fallback for 3D detection")
            
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop the camera pipeline"""
        try:
            self.pipeline.stop()
            print("Camera stopped")
        except Exception as e:
            print(f"Error stopping camera: {e}")
    
    def get_frames(self):
        """Get color frame and estimated depth"""
        try:
            # Get color frame
            frames = self.pipeline.wait_for_frames()
            color_frame = np.asanyarray(frames.get_color_frame().get_data())
            
            # Estimate depth using Depth Anything v2
            depth_map, _ = self.depth_estimator.estimate_depth(color_frame)
            
            return color_frame, depth_map, None
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None, None
    
    def _create_camera_params(self):
        """Create camera parameters"""
        import pyrealsense2 as rs
        
        # Create camera matrix from intrinsics
        camera_matrix = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Create projection matrix
        projection_matrix = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx, 0],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy, 0],
            [0, 0, 1, 0]
        ])
        
        self.camera_params = {
            'camera_matrix': camera_matrix,
            'projection_matrix': projection_matrix,
            'image_width': self.width,
            'image_height': self.height,
            'fx': self.color_intrinsics.fx,
            'fy': self.color_intrinsics.fy,
            'ppx': self.color_intrinsics.ppx,
            'ppy': self.color_intrinsics.ppy,
            'depth_scale': 0.001
        }
    
    def get_camera_params(self):
        """Get camera parameters"""
        return self.camera_params

def main():
    """Main function for hybrid RealSense YOLO-3D."""
    # Configuration variables
    # ===============================================
    
    # Camera settings
    camera_width = 640
    camera_height = 480
    camera_fps = 30
    
    # Output settings
    output_path = "hybrid_realsense_output.mp4"
    
    # Model settings
    yolo_model_size = "nano"
    depth_model_size = "small"
    
    # Device settings
    device = 'cpu'
    
    # Detection settings
    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = None
    
    # Feature toggles
    enable_tracking = True
    enable_bev = True
    enable_3d_visualization = True
    show_depth_map = True
    # ===============================================
    
    print("=== Hybrid RealSense YOLO-3D Object Detection ===")
    print(f"Using device: {device}")
    print("Using color stream + depth estimation fallback")
    
    # Initialize hybrid camera
    print("Initializing hybrid RealSense camera...")
    camera = HybridRealSenseCamera(
        width=camera_width,
        height=camera_height,
        fps=camera_fps
    )
    
    if not camera.start():
        print("Error: Could not start camera")
        return
    
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
        return
    
    # Initialize 3D bounding box estimator (using original)
    print("Initializing 3D bounding box estimator...")
    from bbox3d_utils import BBox3DEstimator, BirdEyeView
    
    # Apply camera parameters
    camera_params = camera.get_camera_params()
    bbox3d_estimator = BBox3DEstimator(
        camera_matrix=camera_params['camera_matrix'],
        projection_matrix=camera_params['projection_matrix']
    )
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        print("Initializing Bird's Eye View...")
        bev = BirdEyeView(scale=60, size=(400, 400))
    
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
        # Check for key press
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
            # Get frames from hybrid camera
            color_frame, depth_map, _ = camera.get_frames()
            
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
            
            # Step 2: Process 3D bounding boxes
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
                        
                        # Get depth at center of bounding box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        depth_value = depth_map[int(center_y), int(center_x)]
                        
                        # Estimate 3D bounding box
                        box_3d = bbox3d_estimator.estimate_3d_box(bbox_2d, depth_value, class_name, object_id)
                        
                        if box_3d['valid']:
                            boxes_3d.append(box_3d)
                            
                            # Draw 3D box on result frame
                            if enable_3d_visualization:
                                result_frame = bbox3d_estimator.draw_3d_box(result_frame, box_3d)
                        
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue
            
            # Step 3: Update Bird's Eye View
            if enable_bev and boxes_3d:
                try:
                    bev_image = bev.update(boxes_3d)
                except Exception as e:
                    print(f"Error updating BEV: {e}")
                    bev_image = np.zeros((400, 400, 3), dtype=np.uint8)
            else:
                bev_image = np.zeros((400, 400, 3), dtype=np.uint8)
            
            # Step 4: Create depth visualization
            if show_depth_map and depth_map is not None:
                try:
                    # Normalize depth for display
                    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                except Exception as e:
                    print(f"Error creating depth colormap: {e}")
                    depth_colormap = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            else:
                depth_colormap = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            
            # Step 5: Create combined visualization
            # Resize images to fit in display
            display_height = 300
            display_width = int(display_height * camera_width / camera_height)
            
            # Resize main frames
            result_display = cv2.resize(result_frame, (display_width, display_height))
            depth_display = cv2.resize(depth_colormap, (display_width, display_height))
            
            # Create combined view
            combined_view = np.zeros((display_height * 2, display_width + 400, 3), dtype=np.uint8)
            
            # Top row: main view and BEV
            combined_view[:display_height, :display_width] = result_display
            combined_view[:display_height, display_width:] = cv2.resize(bev_image, (400, display_height))
            
            # Bottom row: depth map
            combined_view[display_height:, :display_width] = depth_display
            
            # Add labels
            cv2.putText(combined_view, "3D Detection (Hybrid)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, "Bird's Eye View", (display_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, "Estimated Depth", (10, display_height + 30), 
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
            cv2.imshow('Hybrid RealSense YOLO-3D', combined_view)
            
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

if __name__ == "__main__":
    main()
