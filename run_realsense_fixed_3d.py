#!/usr/bin/env python3
"""
RealSense YOLO-3D with Real Depth and Fixed 3D Bounding Boxes
Properly uses bbox3d_utils.py methods and fixes Bird's Eye View
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
from bbox3d_utils import BBox3DEstimator, BirdEyeView

class RealSenseFixed3D:
    """
    RealSense camera with real depth support and fixed 3D bounding boxes
    """
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable both color and depth streams
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        # Alignment for depth-color alignment
        self.align_to_color = rs.align(rs.stream.color)
        
        # Depth filters for better quality
        self.depth_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        
        # Camera parameters
        self.camera_params = None
        self.depth_scale = None
        self.real_depth_available = False
        
    def start(self):
        """Start the camera pipeline"""
        try:
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Get depth stream
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Test if depth stream works
            print("Testing depth stream...")
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            depth_frame = frames.get_depth_frame()
            
            if depth_frame:
                self.real_depth_available = True
                print("✓ Real depth stream working!")
            else:
                print("⚠️ Depth stream not producing frames")
                self.real_depth_available = False
            
            # Create camera parameters
            self._create_camera_params()
            
            print(f"RealSense camera started successfully")
            print(f"Color intrinsics: {self.color_intrinsics}")
            print(f"Real depth available: {self.real_depth_available}")
            
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
        """Get color frame and real depth"""
        try:
            frames = self.pipeline.wait_for_frames()
            
            # Get color frame
            color_frame = np.asanyarray(frames.get_color_frame().get_data())
            
            # Get depth frame
            if self.real_depth_available:
                try:
                    # Align depth to color
                    aligned_frames = self.align_to_color.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if depth_frame:
                        # Apply filters for better quality
                        depth_frame = self.depth_filter.process(depth_frame)
                        depth_frame = self.temporal_filter.process(depth_frame)
                        depth_frame = self.hole_filling_filter.process(depth_frame)
                        
                        # Convert to numpy array and meters
                        depth_data = np.asanyarray(depth_frame.get_data())
                        depth_map = depth_data.astype(np.float32) * self.depth_scale
                        
                        return color_frame, depth_map, "real"
                    else:
                        return color_frame, None, "failed"
                        
                except Exception as e:
                    print(f"Real depth error: {e}")
                    return color_frame, None, "failed"
            else:
                return color_frame, None, "unavailable"
                
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None, None
    
    def get_depth_in_bbox(self, bbox, depth_map):
        """Get average depth value within a bounding box"""
        if depth_map is None:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Clamp coordinates to frame bounds
        x1 = max(0, min(x1, depth_map.shape[1] - 1))
        y1 = max(0, min(y1, depth_map.shape[0] - 1))
        x2 = max(0, min(x2, depth_map.shape[1] - 1))
        y2 = max(0, min(y2, depth_map.shape[0] - 1))
        
        # Extract depth values in the bounding box
        depth_roi = depth_map[y1:y2, x1:x2]
        
        # Filter out invalid depth values
        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi <= 10.0)]
        
        if len(valid_depths) == 0:
            return 0.0
        
        return float(np.mean(valid_depths))
    
    def _create_camera_params(self):
        """Create camera parameters"""
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
            'depth_scale': self.depth_scale if self.depth_scale else 0.001
        }
    
    def get_camera_params(self):
        """Get camera parameters"""
        return self.camera_params

def main():
    print("=== RealSense YOLO-3D with Real Depth and Fixed 3D Boxes ===")
    
    # Configuration
    camera_width = 640
    camera_height = 480
    camera_fps = 30
    
    # Initialize camera
    print("Initializing RealSense camera...")
    camera = RealSenseFixed3D(
        width=camera_width,
        height=camera_height,
        fps=camera_fps
    )
    
    if not camera.start():
        print("Error: Could not start camera")
        return
    
    # Initialize models
    print("Initializing models...")
    try:
        detector = ObjectDetector(model_size="nano", device="cpu")
        camera_params = camera.get_camera_params()
        
        # Use the existing BBox3DEstimator with proper camera parameters
        bbox3d_estimator = BBox3DEstimator(
            camera_matrix=camera_params['camera_matrix'],
            projection_matrix=camera_params['projection_matrix']
        )
        
        # Initialize Bird's Eye View and reset it to show the grid
        bev = BirdEyeView(scale=60, size=(400, 400))
        bev.reset()  # This initializes the grid and background
        
        print("✓ All models initialized")
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        camera.stop()
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
            color_frame, depth_map, depth_type = camera.get_frames()
            
            if color_frame is None:
                continue
            
            # Object detection
            detection_frame, detections = detector.detect(color_frame.copy(), track=True)
            
            # Process 3D boxes using the existing bbox3d_utils methods
            boxes_3d = []
            if detections and depth_map is not None:
                for detection in detections:
                    try:
                        # Handle different detection formats
                        if len(detection) >= 7:  # With tracking
                            x1, y1, x2, y2, conf, class_id, track_id = detection[:7]
                            object_id = int(track_id) if track_id is not None else None
                        elif len(detection) >= 6:  # Without tracking
                            x1, y1, x2, y2, conf, class_id = detection[:6]
                            object_id = None
                        elif len(detection) >= 5:  # Basic format
                            x1, y1, x2, y2, conf = detection[:5]
                            class_id = 0  # Default class
                            object_id = None
                        else:
                            continue
                        
                        class_name = detector.class_names[int(class_id)]
                        bbox_2d = [int(x1), int(y1), int(x2), int(y2)]
                        
                        # Get depth at center of bounding box
                        depth_value = camera.get_depth_in_bbox(bbox_2d, depth_map)
                        
                        # Use the existing estimate_3d_box method
                        box_3d = bbox3d_estimator.estimate_3d_box(bbox_2d, depth_value, class_name, object_id)
                        
                        if box_3d['valid']:
                            # Add depth information to the box
                            box_3d['depth_value'] = depth_value
                            box_3d['depth_method'] = depth_type
                            box_3d['score'] = conf
                            boxes_3d.append(box_3d)
                            
                            # Use the existing draw_box_3d method for proper 3D visualization
                            detection_frame = bbox3d_estimator.draw_box_3d(detection_frame, box_3d)
                        
                    except Exception as e:
                        print(f"Detection error: {e}")
                        continue
            
            # Update Bird's Eye View properly
            # Reset the BEV first to clear previous objects
            bev.reset()
            
            # Draw each object on the BEV
            for box_3d in boxes_3d:
                if box_3d['valid']:
                    bev.draw_box(box_3d)
            
            # Get the updated BEV image
            bev_image = bev.get_image()
            
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
            depth_label = f"Real Depth ({depth_type})" if depth_type else "Depth"
            cv2.putText(combined_view, "3D Detection with Real Depth", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, "Bird's Eye View", (display_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_view, depth_label, (10, display_height + 30), 
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
            cv2.imshow('RealSense YOLO-3D with Real Depth', combined_view)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()
