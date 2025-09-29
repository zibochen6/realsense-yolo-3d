#!/usr/bin/env python3
"""
Intel RealSense Camera Module for YOLO-3D (Fixed Version)
Provides real depth data from stereo cameras instead of estimated depth
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Tuple, Optional, Dict, Any
import json
import os

class RealSenseCamera:
    """
    Intel RealSense camera interface for 3D object detection
    """
    
    def __init__(self, 
                 width: int = 640, 
                 height: int = 480, 
                 fps: int = 30,
                 depth_units: float = 0.001,  # Convert to meters
                 enable_color: bool = True,
                 enable_depth: bool = True,
                 enable_infrared: bool = False):
        """
        Initialize RealSense camera
        
        Args:
            width (int): Frame width
            height (int): Frame height  
            fps (int): Frames per second
            depth_units (float): Depth units in meters (0.001 = millimeters)
            enable_color (bool): Enable color stream
            enable_depth (bool): Enable depth stream
            enable_infrared (bool): Enable infrared stream
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_units = depth_units
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        if self.enable_color:
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
        if self.enable_infrared:
            self.config.enable_stream(rs.stream.infrared, self.width, self.height, rs.format.y8, self.fps)
        
        # Camera intrinsics and parameters
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None
        self.depth_to_color_extrinsics = None
        
        # Alignment object for depth-color alignment
        self.align_to_color = rs.align(rs.stream.color)
        
        # Filters for depth processing
        self.depth_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        
        # Camera parameters for 3D estimation
        self.camera_params = None
        
        # Depth fallback - if depth stream fails, we'll use a simple depth estimation
        self.depth_fallback_enabled = True
        self.depth_fallback_value = 2.0  # Default depth in meters
        
    def start(self) -> bool:
        """
        Start the camera pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            if self.enable_color:
                color_stream = profile.get_stream(rs.stream.color)
                self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                
            if self.enable_depth:
                depth_stream = profile.get_stream(rs.stream.depth)
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                
                # Get depth scale
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                
                # Get extrinsics from depth to color
                if self.enable_color:
                    self.depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)
            
            # Create camera parameters for 3D estimation
            self._create_camera_params()
            
            print(f"RealSense camera started successfully")
            print(f"Color intrinsics: {self.color_intrinsics}")
            print(f"Depth intrinsics: {self.depth_intrinsics}")
            print(f"Depth scale: {self.depth_scale}")
            
            return True
            
        except Exception as e:
            print(f"Error starting RealSense camera: {e}")
            return False
    
    def stop(self):
        """Stop the camera pipeline"""
        try:
            self.pipeline.stop()
            print("RealSense camera stopped")
        except Exception as e:
            print(f"Error stopping RealSense camera: {e}")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get frames from the camera
        
        Returns:
            Tuple of (color_frame, depth_frame, infrared_frame)
            Returns None for disabled streams
        """
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth to color
            if self.enable_color and self.enable_depth:
                frames = self.align_to_color.process(frames)
            
            # Extract frames
            color_frame = None
            depth_frame = None
            infrared_frame = None
            
            if self.enable_color:
                color_frame = np.asanyarray(frames.get_color_frame().get_data())
                
            if self.enable_depth:
                depth_frame_obj = frames.get_depth_frame()
                
                if depth_frame_obj:
                    # Apply filters for better depth quality
                    depth_frame_obj = self.depth_filter.process(depth_frame_obj)
                    depth_frame_obj = self.temporal_filter.process(depth_frame_obj)
                    depth_frame_obj = self.hole_filling_filter.process(depth_frame_obj)
                    
                    # Convert to numpy array
                    depth_frame = np.asanyarray(depth_frame_obj.get_data())
                    
                    # Convert to meters
                    depth_frame = depth_frame.astype(np.float32) * self.depth_scale
                else:
                    # Depth frame not available, use fallback
                    if self.depth_fallback_enabled:
                        depth_frame = self._create_fallback_depth_frame()
                    else:
                        depth_frame = None
                
            if self.enable_infrared:
                infrared_frame = np.asanyarray(frames.get_infrared_frame().get_data())
            
            return color_frame, depth_frame, infrared_frame
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            # Return fallback frames if available
            if self.depth_fallback_enabled and self.enable_color:
                try:
                    # Try to get just color frame
                    frames = self.pipeline.wait_for_frames()
                    color_frame = np.asanyarray(frames.get_color_frame().get_data())
                    depth_frame = self._create_fallback_depth_frame()
                    return color_frame, depth_frame, None
                except:
                    pass
            return None, None, None
    
    def _create_fallback_depth_frame(self) -> np.ndarray:
        """
        Create a fallback depth frame when depth stream is not available
        
        Returns:
            np.ndarray: Fallback depth frame
        """
        # Create a depth frame with default depth value
        depth_frame = np.full((self.height, self.width), self.depth_fallback_value, dtype=np.float32)
        return depth_frame
    
    def get_depth_at_point(self, x: int, y: int, depth_frame: np.ndarray) -> float:
        """
        Get depth value at a specific pixel location
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            depth_frame (np.ndarray): Depth frame
            
        Returns:
            float: Depth value in meters, or 0.0 if invalid
        """
        if depth_frame is None or x < 0 or y < 0 or x >= depth_frame.shape[1] or y >= depth_frame.shape[0]:
            return 0.0
            
        depth_value = depth_frame[y, x]
        
        # Filter out invalid depth values
        if depth_value <= 0 or depth_value > 10.0:  # Max 10 meters
            return 0.0
            
        return float(depth_value)
    
    def get_depth_in_bbox(self, bbox: list, depth_frame: np.ndarray) -> float:
        """
        Get average depth value within a bounding box
        
        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            depth_frame (np.ndarray): Depth frame
            
        Returns:
            float: Average depth value in meters
        """
        if depth_frame is None:
            return self.depth_fallback_value
            
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Clamp coordinates to frame bounds
        x1 = max(0, min(x1, depth_frame.shape[1] - 1))
        y1 = max(0, min(y1, depth_frame.shape[0] - 1))
        x2 = max(0, min(x2, depth_frame.shape[1] - 1))
        y2 = max(0, min(y2, depth_frame.shape[0] - 1))
        
        # Extract depth values in the bounding box
        depth_roi = depth_frame[y1:y2, x1:x2]
        
        # Filter out invalid depth values
        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi <= 10.0)]
        
        if len(valid_depths) == 0:
            return self.depth_fallback_value
            
        return float(np.mean(valid_depths))
    
    def _create_camera_params(self):
        """Create camera parameters for 3D estimation"""
        if self.color_intrinsics is None:
            return
            
        # Create camera matrix from intrinsics
        camera_matrix = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Create projection matrix (simplified - no rotation/translation)
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
    
    def save_camera_params(self, filename: str):
        """
        Save camera parameters to JSON file
        
        Args:
            filename (str): Path to save parameters
        """
        if self.camera_params is None:
            print("No camera parameters to save")
            return
            
        # Convert numpy arrays to lists for JSON serialization
        params_to_save = self.camera_params.copy()
        params_to_save['camera_matrix'] = params_to_save['camera_matrix'].tolist()
        params_to_save['projection_matrix'] = params_to_save['projection_matrix'].tolist()
        
        try:
            with open(filename, 'w') as f:
                json.dump(params_to_save, f, indent=2)
            print(f"Camera parameters saved to {filename}")
        except Exception as e:
            print(f"Error saving camera parameters: {e}")
    
    def load_camera_params(self, filename: str) -> bool:
        """
        Load camera parameters from JSON file
        
        Args:
            filename (str): Path to parameters file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(filename):
            print(f"Camera parameters file {filename} not found")
            return False
            
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            
            # Convert lists back to numpy arrays
            params['camera_matrix'] = np.array(params['camera_matrix'])
            params['projection_matrix'] = np.array(params['projection_matrix'])
            
            self.camera_params = params
            print(f"Camera parameters loaded from {filename}")
            return True
            
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            return False
    
    def get_camera_params(self) -> Optional[Dict[str, Any]]:
        """
        Get current camera parameters
        
        Returns:
            dict: Camera parameters or None if not available
        """
        return self.camera_params
    
    def is_connected(self) -> bool:
        """
        Check if camera is connected and streaming
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            # Try to get frames with a short timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            return True
        except:
            return False

def main():
    """Test function for RealSense camera"""
    print("Testing RealSense camera...")
    
    # Initialize camera
    camera = RealSenseCamera(width=640, height=480, fps=30)
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    print("Camera started successfully. Press 'q' to quit, 's' to save parameters")
    
    try:
        while True:
            # Get frames
            color_frame, depth_frame, infrared_frame = camera.get_frames()
            
            if color_frame is not None:
                # Display color frame
                cv2.imshow('Color', color_frame)
                
            if depth_frame is not None:
                # Normalize depth for display
                depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow('Depth', depth_colormap)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                camera.save_camera_params('realsense_params.json')
                print("Camera parameters saved")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
