#!/usr/bin/env python3
"""
RealSense YOLO-3D with fixed detection parsing and proper 3D cubes
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
from bbox3d_utils import BBox3DEstimator

class RealSenseFixed3D:
    """
    RealSense camera with REAL depth support and fixed detection parsing
    """
    
    def __init__(self, width=640, height=480, fps=6):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        # Alignment for depth-color alignment
        self.align_to_color = rs.align(rs.stream.color)
        
        # Depth filters
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
        """Get color frame and REAL depth"""
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

def draw_proper_3d_cube(image, bbox_2d, depth_value, class_name, color=(0, 255, 0), thickness=2):
    """
    Draw a proper 3D cube based on Ultralytics 3D object detection principles
    """
    try:
        x1, y1, x2, y2 = bbox_2d
        
        # Calculate 3D cube dimensions based on depth
        # Closer objects should have larger 3D effect
        depth_factor = max(0.1, min(1.0, 2.0 - depth_value))  # Inverse relationship with depth
        
        # Calculate offsets for 3D effect
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # 3D offset based on depth and object size
        offset_x = int(width_2d * 0.25 * depth_factor)
        offset_y = int(height_2d * 0.25 * depth_factor)
        
        # Ensure minimum offset for visibility
        offset_x = max(15, min(offset_x, 40))
        offset_y = max(15, min(offset_y, 40))
        
        # Define 8 corners of the 3D cube
        # Front face (2D bounding box)
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)
        
        # Back face (offset by depth)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        
        # Create overlay for transparency effects
        overlay = image.copy()
        
        # Draw the 12 edges of the 3D cube
        # Front face edges
        cv2.rectangle(image, front_tl, front_br, color, thickness)
        
        # Back face edges
        cv2.line(image, back_tl, back_tr, color, thickness)
        cv2.line(image, back_tr, back_br, color, thickness)
        cv2.line(image, back_br, back_bl, color, thickness)
        cv2.line(image, back_bl, back_tl, color, thickness)
        
        # Connecting edges (front to back)
        cv2.line(image, front_tl, back_tl, color, thickness)
        cv2.line(image, front_tr, back_tr, color, thickness)
        cv2.line(image, front_br, back_br, color, thickness)
        cv2.line(image, front_bl, back_bl, color, thickness)
        
        # Fill faces with transparency for better 3D effect
        # Top face
        top_face = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
        cv2.fillPoly(overlay, [top_face], color)
        
        # Right face (darker)
        right_face = np.array([front_tr, front_br, back_br, back_tr], np.int32)
        right_color = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
        cv2.fillPoly(overlay, [right_face], right_color)
        
        # Apply transparency
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Add text labels with background
        text_y = y1 - 10
        text_color = (255, 255, 255)  # White text
        
        # Class name
        text = class_name
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, text_y - text_h - 5), (x1 + text_w, text_y + 5), (0, 0, 0), -1)
        cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        text_y -= 20
        
        # Depth
        text = f"D:{depth_value:.2f}m"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, text_y - text_h - 5), (x1 + text_w, text_y + 5), (0, 0, 0), -1)
        cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # Draw ground contact point
        ground_y = y2 + int((y2 - y1) * 0.3)
        cv2.line(image, (int((x1 + x2) / 2), y2), (int((x1 + x2) / 2), ground_y), color, thickness)
        cv2.circle(image, (int((x1 + x2) / 2), ground_y), thickness * 2, color, -1)
        
        return image
        
    except Exception as e:
        print(f"Error drawing 3D cube: {e}")
        return image

def create_bev_with_objects(boxes_3d, size=(400, 400)):
    """Create Bird's Eye View with objects"""
    bev_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Draw grid
    for x in range(0, size[0], 50):
        cv2.line(bev_image, (x, 0), (x, size[1]), (50, 50, 50), 1)
    for y in range(0, size[1], 50):
        cv2.line(bev_image, (0, y), (size[0], y), (50, 50, 50), 1)
    
    # Draw camera position
    camera_x = size[0] // 2
    camera_y = size[1] - 20
    cv2.circle(bev_image, (camera_x, camera_y), 5, (255, 255, 255), -1)
    cv2.putText(bev_image, "Camera", (camera_x - 20, camera_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw distance markers
    for dist in [1, 2, 3, 4, 5]:
        y = camera_y - int(dist * 40)  # 40 pixels per meter
        if y > 20:
            cv2.line(bev_image, (camera_x - 5, y), (camera_x + 5, y), (120, 120, 120), 1)
            cv2.putText(bev_image, f"{dist}m", (camera_x + 10, y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    # Draw objects
    for box_3d in boxes_3d:
        if box_3d['valid']:
            center = box_3d['center']
            depth = box_3d.get('depth_value', 1.0)
            class_name = box_3d['class_name']
            
            # Convert 3D position to BEV coordinates
            # X is horizontal offset from center, Y is depth (forward)
            x = int(camera_x + center[0] * 40)  # Scale factor for horizontal
            y = int(camera_y - depth * 40)  # Depth is forward (upward in BEV)
            
            if 0 <= x < size[0] and 0 <= y < size[1]:
                # Color based on class
                if 'person' in class_name:
                    color = (0, 255, 0)  # Green
                elif 'car' in class_name or 'vehicle' in class_name:
                    color = (0, 0, 255)  # Red
                elif 'truck' in class_name or 'bus' in class_name:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (255, 0, 0)  # Blue
                
                # Draw object as a circle
                cv2.circle(bev_image, (x, y), 8, color, -1)
                cv2.circle(bev_image, (x, y), 8, (255, 255, 255), 2)  # White border
                
                # Draw class name
                cv2.putText(bev_image, class_name[:3], (x-10, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # Draw distance line from camera to object
                cv2.line(bev_image, (camera_x, camera_y), (x, y), (70, 70, 70), 1)
    
    return bev_image

def main():
    print("=== RealSense YOLO-3D with Fixed Detection and 3D Cubes ===")
    
    # Configuration
    camera_width = 640
    camera_height = 480
    camera_fps = 6
    
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
        
        # Use the existing BBox3DEstimator
        bbox3d_estimator = BBox3DEstimator(
            camera_matrix=camera_params['camera_matrix'],
            projection_matrix=camera_params['projection_matrix']
        )
        
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
            
            # Process 3D boxes using REAL depth
            boxes_3d = []
            if detections and depth_map is not None:
                for detection in detections:
                    try:
                        # FIXED: Parse the correct detection format
                        # Format: [[x1, y1, x2, y2], conf, class_id, track_id]
                        bbox_coords = detection[0]  # [x1, y1, x2, y2]
                        conf = detection[1]
                        class_id = detection[2]
                        track_id = detection[3] if len(detection) > 3 else None
                        
                        x1, y1, x2, y2 = bbox_coords
                        object_id = int(track_id) if track_id is not None else None
                        
                        class_name = detector.class_names[int(class_id)]
                        bbox_2d = [int(x1), int(y1), int(x2), int(y2)]
                        
                        # Get REAL depth at center of bounding box
                        depth_value = camera.get_depth_in_bbox(bbox_2d, depth_map)
                        
                        # Draw 3D cube directly
                        color = (0, 255, 0) if 'person' in class_name else (255, 0, 0)
                        detection_frame = draw_proper_3d_cube(detection_frame, bbox_2d, depth_value, class_name, color)
                        
                        # Try 3D box estimation for BEV
                        box_3d = bbox3d_estimator.estimate_3d_box(bbox_2d, depth_value, class_name, object_id)
                        
                        if box_3d['valid']:
                            # Add depth information to the box
                            box_3d['depth_value'] = depth_value
                            box_3d['depth_method'] = depth_type
                            box_3d['score'] = conf
                            boxes_3d.append(box_3d)
                        
                    except Exception as e:
                        print(f"Detection error: {e}")
                        continue
            
            # Create Bird's Eye View with objects
            bev_image = create_bev_with_objects(boxes_3d)
            
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
            depth_label = f"REAL Depth ({depth_type})" if depth_type else "Depth"
            cv2.putText(combined_view, "3D Detection with REAL Depth", (10, 30), 
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
            cv2.imshow('RealSense YOLO-3D with Fixed Detection', combined_view)
            
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
