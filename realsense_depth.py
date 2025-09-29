#!/usr/bin/env python3
"""
RealSense Depth Module for YOLO-3D
Uses real depth data from Intel RealSense stereo cameras
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from realsense_camera import RealSenseCamera

class RealSenseDepthEstimator:
    """
    Real depth estimation using Intel RealSense stereo cameras
    """
    
    def __init__(self, camera: RealSenseCamera):
        """
        Initialize the RealSense depth estimator
        
        Args:
            camera (RealSenseCamera): RealSense camera instance
        """
        self.camera = camera
        self.depth_frame = None
        self.color_frame = None
        
    def estimate_depth(self, color_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get real depth data from RealSense camera
        
        Args:
            color_frame (np.ndarray): Color frame (not used, but kept for compatibility)
            
        Returns:
            Tuple of (depth_map, color_frame): Real depth map and color frame
        """
        # Get frames from RealSense camera
        color_frame, depth_frame, _ = self.camera.get_frames()
        
        if color_frame is None or depth_frame is None:
            # Return empty frames if camera failed
            height, width = 480, 640  # Default size
            return np.zeros((height, width), dtype=np.float32), np.zeros((height, width, 3), dtype=np.uint8)
        
        # Store frames for other methods
        self.depth_frame = depth_frame
        self.color_frame = color_frame
        
        return depth_frame, color_frame
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """
        Get depth value at a specific pixel location
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth value in meters
        """
        return self.camera.get_depth_at_point(x, y, self.depth_frame)
    
    def get_depth_in_bbox(self, bbox: List[int]) -> float:
        """
        Get average depth value within a bounding box
        
        Args:
            bbox (List[int]): Bounding box [x1, y1, x2, y2]
            
        Returns:
            float: Average depth value in meters
        """
        return self.camera.get_depth_in_bbox(bbox, self.depth_frame)
    
    def get_depth_map(self) -> Optional[np.ndarray]:
        """
        Get the current depth map
        
        Returns:
            np.ndarray: Current depth map or None
        """
        return self.depth_frame
    
    def get_color_frame(self) -> Optional[np.ndarray]:
        """
        Get the current color frame
        
        Returns:
            np.ndarray: Current color frame or None
        """
        return self.color_frame
    
    def create_depth_colormap(self, depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a colorized depth map for visualization
        
        Args:
            depth_map (np.ndarray): Depth map to colorize (uses current if None)
            
        Returns:
            np.ndarray: Colorized depth map
        """
        if depth_map is None:
            depth_map = self.depth_frame
            
        if depth_map is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Normalize depth for display (0-10 meters range)
        depth_normalized = np.clip(depth_map, 0, 10.0)
        depth_normalized = (depth_normalized / 10.0 * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colormap
    
    def filter_depth_map(self, depth_map: np.ndarray, 
                        min_depth: float = 0.1, 
                        max_depth: float = 10.0) -> np.ndarray:
        """
        Filter depth map to remove invalid values
        
        Args:
            depth_map (np.ndarray): Input depth map
            min_depth (float): Minimum valid depth in meters
            max_depth (float): Maximum valid depth in meters
            
        Returns:
            np.ndarray: Filtered depth map
        """
        filtered_depth = depth_map.copy()
        
        # Set invalid depths to 0
        invalid_mask = (filtered_depth < min_depth) | (filtered_depth > max_depth)
        filtered_depth[invalid_mask] = 0.0
        
        return filtered_depth
    
    def get_depth_statistics(self, depth_map: Optional[np.ndarray] = None) -> dict:
        """
        Get statistics about the depth map
        
        Args:
            depth_map (np.ndarray): Depth map to analyze (uses current if None)
            
        Returns:
            dict: Depth statistics
        """
        if depth_map is None:
            depth_map = self.depth_frame
            
        if depth_map is None:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'valid_pixels': 0}
        
        # Filter out invalid depths
        valid_depths = depth_map[(depth_map > 0) & (depth_map <= 10.0)]
        
        if len(valid_depths) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'valid_pixels': 0}
        
        return {
            'mean': float(np.mean(valid_depths)),
            'std': float(np.std(valid_depths)),
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'valid_pixels': len(valid_depths),
            'total_pixels': depth_map.size
        }

def main():
    """Test function for RealSense depth estimator"""
    print("Testing RealSense depth estimator...")
    
    # Initialize camera and depth estimator
    camera = RealSenseCamera(width=640, height=480, fps=30)
    depth_estimator = RealSenseDepthEstimator(camera)
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    print("Depth estimator started. Press 'q' to quit")
    
    try:
        while True:
            # Get depth and color frames
            depth_map, color_frame = depth_estimator.estimate_depth(None)
            
            if color_frame is not None:
                cv2.imshow('Color', color_frame)
            
            if depth_map is not None:
                # Show colorized depth
                depth_colormap = depth_estimator.create_depth_colormap()
                cv2.imshow('Depth', depth_colormap)
                
                # Print depth statistics
                stats = depth_estimator.get_depth_statistics()
                print(f"Depth stats - Mean: {stats['mean']:.2f}m, "
                      f"Min: {stats['min']:.2f}m, Max: {stats['max']:.2f}m, "
                      f"Valid: {stats['valid_pixels']}/{stats['total_pixels']}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
