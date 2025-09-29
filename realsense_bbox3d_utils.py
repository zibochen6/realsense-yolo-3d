#!/usr/bin/env python3
"""
RealSense 3D Bounding Box Utilities for YOLO-3D
Enhanced 3D bounding box estimation using real depth data from Intel RealSense
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import math
from typing import List, Dict, Tuple, Optional
from realsense_camera import RealSenseCamera

# Average dimensions for common objects (height, width, length) in meters
DEFAULT_DIMS = {
    'car': np.array([1.52, 1.64, 3.85]),
    'truck': np.array([3.07, 2.63, 11.17]),
    'bus': np.array([3.07, 2.63, 11.17]),
    'motorcycle': np.array([1.50, 0.90, 2.20]),
    'bicycle': np.array([1.40, 0.70, 1.80]),
    'person': np.array([1.75, 0.60, 0.60]),
    'dog': np.array([0.80, 0.50, 1.10]),
    'cat': np.array([0.40, 0.30, 0.70]),
    'potted plant': np.array([0.80, 0.40, 0.40]),
    'plant': np.array([0.80, 0.40, 0.40]),
    'chair': np.array([0.80, 0.60, 0.60]),
    'sofa': np.array([0.80, 0.85, 2.00]),
    'table': np.array([0.75, 1.20, 1.20]),
    'bed': np.array([0.60, 1.50, 2.00]),
    'tv': np.array([0.80, 0.15, 1.20]),
    'laptop': np.array([0.02, 0.25, 0.35]),
    'keyboard': np.array([0.03, 0.15, 0.45]),
    'mouse': np.array([0.03, 0.06, 0.10]),
    'book': np.array([0.03, 0.20, 0.15]),
    'bottle': np.array([0.25, 0.10, 0.10]),
    'cup': np.array([0.10, 0.08, 0.08]),
    'vase': np.array([0.30, 0.15, 0.15])
}

class RealSenseBBox3DEstimator:
    """
    3D bounding box estimation using real depth data from Intel RealSense
    """
    
    def __init__(self, camera: RealSenseCamera, class_dims: Optional[Dict] = None):
        """
        Initialize the RealSense 3D bounding box estimator
        
        Args:
            camera (RealSenseCamera): RealSense camera instance
            class_dims (dict): Dictionary mapping class names to dimensions (height, width, length)
        """
        self.camera = camera
        self.dims = class_dims if class_dims is not None else DEFAULT_DIMS
        
        # Get camera parameters
        self.camera_params = camera.get_camera_params()
        if self.camera_params is None:
            raise ValueError("Camera parameters not available. Make sure camera is started.")
        
        self.K = self.camera_params['camera_matrix']
        self.P = self.camera_params['projection_matrix']
        
        # Initialize Kalman filters for tracking 3D boxes
        self.kf_trackers = {}
        
        # Store history of 3D boxes for filtering
        self.box_history = defaultdict(list)
        self.max_history = 5
        
        # Depth filtering parameters
        self.min_depth = 0.1  # Minimum valid depth in meters
        self.max_depth = 10.0  # Maximum valid depth in meters
        
    def estimate_3d_box(self, bbox_2d: List[int], class_name: str, 
                       object_id: Optional[int] = None) -> Dict:
        """
        Estimate 3D bounding box from 2D bounding box and real depth data
        
        Args:
            bbox_2d (List[int]): 2D bounding box [x1, y1, x2, y2]
            class_name (str): Class name of the object
            object_id (int): Object ID for tracking (None for no tracking)
            
        Returns:
            dict: 3D bounding box parameters
        """
        # Get 2D box center and dimensions
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Get real depth at the center of the bounding box
        depth_value = self.camera.get_depth_in_bbox(bbox_2d)
        
        if depth_value <= 0 or depth_value > self.max_depth:
            # Fallback: try to get depth at center point
            depth_value = self.camera.get_depth_at_point(int(center_x), int(center_y))
            
        if depth_value <= 0 or depth_value > self.max_depth:
            # No valid depth available
            return self._create_invalid_3d_box(bbox_2d, class_name)
        
        # Get dimensions for the class
        if class_name.lower() in self.dims:
            dimensions = self.dims[class_name.lower()].copy()
        else:
            # Use default car dimensions if class not found
            dimensions = self.dims['car'].copy()
        
        # Adjust dimensions based on 2D box size and depth
        dimensions = self._adjust_dimensions_by_depth(dimensions, depth_value, width_2d, height_2d)
        
        # Calculate 3D position
        # Convert 2D center to 3D world coordinates
        center_3d = self._pixel_to_3d(center_x, center_y, depth_value)
        
        if center_3d is None:
            return self._create_invalid_3d_box(bbox_2d, class_name)
        
        # Create 3D bounding box
        box_3d = {
            'center': center_3d,
            'dimensions': dimensions,
            'rotation': 0.0,  # Assume no rotation for now
            'class_name': class_name,
            'confidence': 1.0,  # Real depth gives us high confidence
            'depth': depth_value,
            'bbox_2d': bbox_2d,
            'valid': True
        }
        
        # Apply Kalman filtering if tracking is enabled
        if object_id is not None:
            box_3d = self._apply_kalman_filter(box_3d, object_id)
        
        return box_3d
    
    def _pixel_to_3d(self, x: float, y: float, depth: float) -> Optional[np.ndarray]:
        """
        Convert pixel coordinates to 3D world coordinates
        
        Args:
            x (float): X pixel coordinate
            y (float): Y pixel coordinate
            depth (float): Depth value in meters
            
        Returns:
            np.ndarray: 3D coordinates [x, y, z] or None if invalid
        """
        if depth <= 0:
            return None
        
        # Get camera intrinsics
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # Convert to 3D coordinates
        x_3d = (x - cx) * depth / fx
        y_3d = (y - cy) * depth / fy
        z_3d = depth
        
        return np.array([x_3d, y_3d, z_3d])
    
    def _adjust_dimensions_by_depth(self, dimensions: np.ndarray, depth: float, 
                                  width_2d: float, height_2d: float) -> np.ndarray:
        """
        Adjust object dimensions based on depth and 2D box size
        
        Args:
            dimensions (np.ndarray): Base dimensions [height, width, length]
            depth (float): Depth value in meters
            width_2d (float): 2D bounding box width
            height_2d (float): 2D bounding box height
            
        Returns:
            np.ndarray: Adjusted dimensions
        """
        adjusted_dims = dimensions.copy()
        
        # Get camera intrinsics for pixel-to-meter conversion
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        
        # Calculate expected 2D size based on 3D dimensions and depth
        expected_width_2d = (dimensions[1] * fx) / depth  # width in 3D -> width in 2D
        expected_height_2d = (dimensions[0] * fy) / depth  # height in 3D -> height in 2D
        
        # Calculate scaling factors
        width_scale = width_2d / expected_width_2d if expected_width_2d > 0 else 1.0
        height_scale = height_2d / expected_height_2d if expected_height_2d > 0 else 1.0
        
        # Use average scale, but limit the adjustment
        scale = (width_scale + height_scale) / 2
        scale = np.clip(scale, 0.5, 2.0)  # Limit scaling to reasonable range
        
        # Apply scaling to dimensions
        adjusted_dims *= scale
        
        return adjusted_dims
    
    def _create_invalid_3d_box(self, bbox_2d: List[int], class_name: str) -> Dict:
        """
        Create an invalid 3D bounding box when depth is not available
        
        Args:
            bbox_2d (List[int]): 2D bounding box
            class_name (str): Class name
            
        Returns:
            dict: Invalid 3D bounding box
        """
        return {
            'center': np.array([0, 0, 0]),
            'dimensions': np.array([0, 0, 0]),
            'rotation': 0.0,
            'class_name': class_name,
            'confidence': 0.0,
            'depth': 0.0,
            'bbox_2d': bbox_2d,
            'valid': False
        }
    
    def _apply_kalman_filter(self, box_3d: Dict, object_id: int) -> Dict:
        """
        Apply Kalman filtering for 3D box tracking
        
        Args:
            box_3d (dict): 3D bounding box
            object_id (int): Object ID
            
        Returns:
            dict: Filtered 3D bounding box
        """
        if object_id not in self.kf_trackers:
            # Initialize Kalman filter for new object
            kf = KalmanFilter(dim_x=6, dim_z=3)  # 3D position + velocity
            
            # State transition matrix (constant velocity model)
            kf.F = np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            
            # Measurement matrix
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])
            
            # Process noise
            kf.Q *= 0.1
            
            # Measurement noise
            kf.R *= 1.0
            
            # Initial state
            kf.x = np.array([box_3d['center'][0], box_3d['center'][1], box_3d['center'][2], 0, 0, 0])
            kf.P *= 100
            
            self.kf_trackers[object_id] = kf
        
        # Update Kalman filter
        kf = self.kf_trackers[object_id]
        kf.predict()
        kf.update(box_3d['center'])
        
        # Use filtered position
        box_3d['center'] = kf.x[:3]
        
        return box_3d
    
    def draw_3d_box(self, frame: np.ndarray, box_3d: Dict, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw 3D bounding box on the frame
        
        Args:
            frame (np.ndarray): Input frame
            box_3d (dict): 3D bounding box
            color (Tuple[int, int, int]): Box color (B, G, R)
            
        Returns:
            np.ndarray: Frame with drawn 3D box
        """
        if not box_3d['valid']:
            return frame
        
        # Get 3D box corners
        corners_3d = self._get_3d_box_corners(box_3d)
        
        # Project 3D corners to 2D
        corners_2d = []
        for corner in corners_3d:
            pixel = self._3d_to_pixel(corner)
            if pixel is not None:
                corners_2d.append(pixel)
        
        if len(corners_2d) != 8:
            return frame
        
        # Draw 3D box edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        for edge in edges:
            pt1 = tuple(map(int, corners_2d[edge[0]]))
            pt2 = tuple(map(int, corners_2d[edge[1]]))
            cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw class label and depth
        center_2d = self._3d_to_pixel(box_3d['center'])
        if center_2d is not None:
            label = f"{box_3d['class_name']}: {box_3d['depth']:.2f}m"
            cv2.putText(frame, label, (int(center_2d[0]), int(center_2d[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _get_3d_box_corners(self, box_3d: Dict) -> List[np.ndarray]:
        """
        Get 8 corners of the 3D bounding box
        
        Args:
            box_3d (dict): 3D bounding box
            
        Returns:
            List[np.ndarray]: List of 8 corner points
        """
        center = box_3d['center']
        dims = box_3d['dimensions']  # [height, width, length]
        rotation = box_3d['rotation']
        
        # Half dimensions
        h_half = dims[0] / 2
        w_half = dims[1] / 2
        l_half = dims[2] / 2
        
        # Create rotation matrix
        R_matrix = R.from_euler('z', rotation).as_matrix()
        
        # Define 8 corners in local coordinate system
        corners_local = np.array([
            [-l_half, -w_half, -h_half],  # Bottom-left-back
            [l_half, -w_half, -h_half],   # Bottom-right-back
            [l_half, w_half, -h_half],    # Bottom-right-front
            [-l_half, w_half, -h_half],   # Bottom-left-front
            [-l_half, -w_half, h_half],   # Top-left-back
            [l_half, -w_half, h_half],    # Top-right-back
            [l_half, w_half, h_half],     # Top-right-front
            [-l_half, w_half, h_half]     # Top-left-front
        ])
        
        # Rotate and translate corners
        corners_3d = []
        for corner in corners_local:
            rotated_corner = R_matrix @ corner
            world_corner = center + rotated_corner
            corners_3d.append(world_corner)
        
        return corners_3d
    
    def _3d_to_pixel(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert 3D world coordinates to pixel coordinates
        
        Args:
            point_3d (np.ndarray): 3D coordinates [x, y, z]
            
        Returns:
            np.ndarray: Pixel coordinates [x, y] or None if invalid
        """
        if point_3d[2] <= 0:  # Invalid depth
            return None
        
        # Get camera intrinsics
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # Project to 2D
        x_2d = (point_3d[0] * fx / point_3d[2]) + cx
        y_2d = (point_3d[1] * fy / point_3d[2]) + cy
        
        return np.array([x_2d, y_2d])

class RealSenseBirdEyeView:
    """
    Bird's Eye View visualization for RealSense 3D data
    """
    
    def __init__(self, scale: int = 60, size: Tuple[int, int] = (300, 300)):
        """
        Initialize Bird's Eye View
        
        Args:
            scale (int): Scale factor for visualization
            size (Tuple[int, int]): Size of the BEV image
        """
        self.scale = scale
        self.size = size
        self.bev_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
    def update(self, boxes_3d: List[Dict], camera_height: float = 1.65) -> np.ndarray:
        """
        Update Bird's Eye View with 3D boxes
        
        Args:
            boxes_3d (List[Dict]): List of 3D bounding boxes
            camera_height (float): Camera height above ground
            
        Returns:
            np.ndarray: Updated BEV image
        """
        # Clear previous image
        self.bev_image.fill(0)
        
        # Draw grid
        self._draw_grid()
        
        # Draw camera position
        camera_x = self.size[0] // 2
        camera_y = self.size[1] - 20
        cv2.circle(self.bev_image, (camera_x, camera_y), 5, (255, 255, 255), -1)
        cv2.putText(self.bev_image, "Camera", (camera_x - 20, camera_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw 3D boxes
        for box_3d in boxes_3d:
            if box_3d['valid']:
                self._draw_box_bev(box_3d, camera_height)
        
        return self.bev_image
    
    def _draw_grid(self):
        """Draw grid lines on BEV"""
        # Vertical lines
        for x in range(0, self.size[0], 50):
            cv2.line(self.bev_image, (x, 0), (x, self.size[1]), (50, 50, 50), 1)
        
        # Horizontal lines
        for y in range(0, self.size[1], 50):
            cv2.line(self.bev_image, (0, y), (self.size[0], y), (50, 50, 50), 1)
    
    def _draw_box_bev(self, box_3d: Dict, camera_height: float):
        """
        Draw a 3D box on Bird's Eye View
        
        Args:
            box_3d (dict): 3D bounding box
            camera_height (float): Camera height above ground
        """
        center = box_3d['center']
        dims = box_3d['dimensions']
        
        # Convert 3D position to BEV coordinates
        # X: left-right, Z: forward-backward
        bev_x = int(self.size[0] // 2 + center[0] * self.scale)
        bev_y = int(self.size[1] - 20 - center[2] * self.scale)  # Z is forward
        
        # Box dimensions in BEV
        box_width = int(dims[1] * self.scale)  # width
        box_length = int(dims[2] * self.scale)  # length
        
        # Draw box rectangle
        pt1 = (bev_x - box_width // 2, bev_y - box_length // 2)
        pt2 = (bev_x + box_width // 2, bev_y + box_length // 2)
        
        # Color based on class
        color = self._get_class_color(box_3d['class_name'])
        cv2.rectangle(self.bev_image, pt1, pt2, color, 2)
        
        # Draw class label
        label = f"{box_3d['class_name']}"
        cv2.putText(self.bev_image, label, (pt1[0], pt1[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for class"""
        colors = {
            'car': (0, 255, 0),      # Green
            'person': (255, 0, 0),   # Blue
            'truck': (0, 255, 255),  # Yellow
            'bus': (0, 255, 255),    # Yellow
            'motorcycle': (255, 0, 255),  # Magenta
            'bicycle': (255, 0, 255),     # Magenta
        }
        return colors.get(class_name.lower(), (128, 128, 128))  # Gray default

def main():
    """Test function for RealSense 3D bounding box utilities"""
    print("Testing RealSense 3D bounding box utilities...")
    
    # Initialize camera and 3D estimator
    camera = RealSenseCamera(width=640, height=480, fps=30)
    bbox3d_estimator = RealSenseBBox3DEstimator(camera)
    bev = RealSenseBirdEyeView(scale=60, size=(300, 300))
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    print("3D bounding box estimator started. Press 'q' to quit")
    
    try:
        while True:
            # Get frames
            color_frame, depth_frame, _ = camera.get_frames()
            
            if color_frame is not None:
                # Simulate a detection (you would get this from YOLO)
                # For testing, create a fake detection in the center
                h, w = color_frame.shape[:2]
                fake_bbox = [w//4, h//4, 3*w//4, 3*h//4]
                fake_class = 'person'
                
                # Estimate 3D box
                box_3d = bbox3d_estimator.estimate_3d_box(fake_bbox, fake_class)
                
                # Draw 3D box
                result_frame = bbox3d_estimator.draw_3d_box(color_frame, box_3d)
                
                # Update BEV
                bev_image = bev.update([box_3d])
                
                # Display results
                cv2.imshow('Color with 3D Box', result_frame)
                cv2.imshow('Bird\'s Eye View', bev_image)
            
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
