#!/usr/bin/env python3
"""
RealSense Camera Calibration Utility
Helps calibrate and save camera parameters for optimal 3D object detection
"""

import numpy as np
import cv2
import json
import os
from realsense_camera import RealSenseCamera
from realsense_depth import RealSenseDepthEstimator

class RealSenseCalibrator:
    """
    RealSense camera calibration utility
    """
    
    def __init__(self, camera: RealSenseCamera):
        """
        Initialize calibrator
        
        Args:
            camera (RealSenseCamera): RealSense camera instance
        """
        self.camera = camera
        self.depth_estimator = RealSenseDepthEstimator(camera)
        
        # Calibration parameters
        self.calibration_data = {
            'depth_offset': 0.0,  # Depth offset correction
            'depth_scale_factor': 1.0,  # Depth scale correction
            'ground_plane_normal': np.array([0, 1, 0]),  # Ground plane normal
            'camera_height': 1.65,  # Camera height above ground
            'depth_quality_threshold': 0.8,  # Minimum depth quality
        }
    
    def calibrate_depth_accuracy(self, num_samples: int = 100) -> dict:
        """
        Calibrate depth accuracy using known distances
        
        Args:
            num_samples (int): Number of samples to collect
            
        Returns:
            dict: Calibration results
        """
        print("Starting depth accuracy calibration...")
        print("Place objects at known distances (0.5m, 1.0m, 1.5m, 2.0m, 3.0m)")
        print("Press 'c' to capture sample, 'q' to quit calibration")
        
        known_distances = [0.5, 1.0, 1.5, 2.0, 3.0]
        measured_distances = []
        actual_distances = []
        
        sample_count = 0
        
        try:
            while sample_count < num_samples:
                # Get frames
                color_frame, depth_frame, _ = self.camera.get_frames()
                
                if color_frame is not None and depth_frame is not None:
                    # Display frames
                    cv2.imshow('Calibration - Color', color_frame)
                    
                    # Create depth visualization
                    depth_colormap = self.depth_estimator.create_depth_colormap(depth_frame)
                    cv2.imshow('Calibration - Depth', depth_colormap)
                    
                    # Get center depth
                    h, w = depth_frame.shape
                    center_depth = self.camera.get_depth_at_point(w//2, h//2)
                    
                    # Display current depth
                    cv2.putText(color_frame, f"Center Depth: {center_depth:.3f}m", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(color_frame, f"Samples: {sample_count}/{num_samples}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(color_frame, "Press 'c' to capture, 'q' to quit", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Calibration - Color', color_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if center_depth > 0:
                        # Ask user for actual distance
                        print(f"Measured depth: {center_depth:.3f}m")
                        try:
                            actual_dist = float(input("Enter actual distance in meters: "))
                            measured_distances.append(center_depth)
                            actual_distances.append(actual_dist)
                            sample_count += 1
                            print(f"Sample {sample_count} recorded: {center_depth:.3f}m -> {actual_dist:.3f}m")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
        
        except KeyboardInterrupt:
            print("Calibration interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
        
        # Calculate calibration parameters
        if len(measured_distances) > 0:
            measured_distances = np.array(measured_distances)
            actual_distances = np.array(actual_distances)
            
            # Linear regression to find scale and offset
            A = np.vstack([measured_distances, np.ones(len(measured_distances))]).T
            scale, offset = np.linalg.lstsq(A, actual_distances, rcond=None)[0]
            
            # Calculate accuracy metrics
            corrected_distances = scale * measured_distances + offset
            errors = np.abs(corrected_distances - actual_distances)
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            
            calibration_results = {
                'depth_scale_factor': float(scale),
                'depth_offset': float(offset),
                'mean_error': float(mean_error),
                'std_error': float(std_error),
                'max_error': float(max_error),
                'num_samples': len(measured_distances),
                'measured_distances': measured_distances.tolist(),
                'actual_distances': actual_distances.tolist(),
                'corrected_distances': corrected_distances.tolist()
            }
            
            print(f"\nCalibration Results:")
            print(f"Scale Factor: {scale:.4f}")
            print(f"Offset: {offset:.4f}")
            print(f"Mean Error: {mean_error:.4f}m")
            print(f"Std Error: {std_error:.4f}m")
            print(f"Max Error: {max_error:.4f}m")
            
            return calibration_results
        else:
            print("No calibration data collected")
            return {}
    
    def calibrate_ground_plane(self, num_samples: int = 50) -> dict:
        """
        Calibrate ground plane detection
        
        Args:
            num_samples (int): Number of samples to collect
            
        Returns:
            dict: Ground plane calibration results
        """
        print("Starting ground plane calibration...")
        print("Point camera at a flat ground surface")
        print("Press 'c' to capture sample, 'q' to quit calibration")
        
        ground_points = []
        sample_count = 0
        
        try:
            while sample_count < num_samples:
                # Get frames
                color_frame, depth_frame, _ = self.camera.get_frames()
                
                if color_frame is not None and depth_frame is not None:
                    # Display frames
                    cv2.imshow('Ground Plane Calibration - Color', color_frame)
                    
                    # Create depth visualization
                    depth_colormap = self.depth_estimator.create_depth_colormap(depth_frame)
                    cv2.imshow('Ground Plane Calibration - Depth', depth_colormap)
                    
                    # Sample points from the center region
                    h, w = depth_frame.shape
                    center_region = depth_frame[h//3:2*h//3, w//3:2*w//3]
                    
                    # Get valid depth points
                    valid_depths = center_region[center_region > 0]
                    if len(valid_depths) > 0:
                        avg_depth = np.mean(valid_depths)
                        cv2.putText(color_frame, f"Avg Depth: {avg_depth:.3f}m", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.putText(color_frame, f"Samples: {sample_count}/{num_samples}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(color_frame, "Press 'c' to capture, 'q' to quit", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Ground Plane Calibration - Color', color_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if len(valid_depths) > 0:
                        # Sample multiple points from the center region
                        for i in range(10):  # Sample 10 points per capture
                            x = np.random.randint(w//3, 2*w//3)
                            y = np.random.randint(h//3, 2*h//3)
                            depth = self.camera.get_depth_at_point(x, y)
                            
                            if depth > 0:
                                # Convert to 3D coordinates
                                point_3d = self._pixel_to_3d(x, y, depth)
                                if point_3d is not None:
                                    ground_points.append(point_3d)
                        
                        sample_count += 1
                        print(f"Ground plane sample {sample_count} collected")
        
        except KeyboardInterrupt:
            print("Ground plane calibration interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
        
        # Calculate ground plane
        if len(ground_points) > 10:
            ground_points = np.array(ground_points)
            
            # Fit plane using SVD
            centroid = np.mean(ground_points, axis=0)
            centered_points = ground_points - centroid
            U, S, Vt = np.linalg.svd(centered_points)
            normal = Vt[-1]  # Last row of Vt is the normal
            
            # Ensure normal points upward
            if normal[1] < 0:
                normal = -normal
            
            # Calculate camera height (distance from camera to ground plane)
            camera_height = np.abs(np.dot(centroid, normal))
            
            calibration_results = {
                'ground_plane_normal': normal.tolist(),
                'ground_plane_centroid': centroid.tolist(),
                'camera_height': float(camera_height),
                'num_points': len(ground_points)
            }
            
            print(f"\nGround Plane Calibration Results:")
            print(f"Normal: {normal}")
            print(f"Centroid: {centroid}")
            print(f"Camera Height: {camera_height:.3f}m")
            
            return calibration_results
        else:
            print("Insufficient ground plane data collected")
            return {}
    
    def _pixel_to_3d(self, x: float, y: float, depth: float) -> np.ndarray:
        """Convert pixel to 3D coordinates"""
        camera_params = self.camera.get_camera_params()
        if camera_params is None:
            return None
        
        K = camera_params['camera_matrix']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x_3d = (x - cx) * depth / fx
        y_3d = (y - cy) * depth / fy
        z_3d = depth
        
        return np.array([x_3d, y_3d, z_3d])
    
    def save_calibration(self, filename: str):
        """
        Save calibration data to file
        
        Args:
            filename (str): Path to save calibration data
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            print(f"Calibration data saved to {filename}")
        except Exception as e:
            print(f"Error saving calibration data: {e}")
    
    def load_calibration(self, filename: str) -> bool:
        """
        Load calibration data from file
        
        Args:
            filename (str): Path to calibration data file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found")
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            if 'ground_plane_normal' in data:
                data['ground_plane_normal'] = np.array(data['ground_plane_normal'])
            
            self.calibration_data.update(data)
            print(f"Calibration data loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False

def main():
    """Main calibration function"""
    print("=== RealSense Camera Calibration ===")
    
    # Initialize camera
    camera = RealSenseCamera(width=640, height=480, fps=30)
    calibrator = RealSenseCalibrator(camera)
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    print("Camera started successfully")
    print("Choose calibration option:")
    print("1. Depth accuracy calibration")
    print("2. Ground plane calibration")
    print("3. Full calibration (both)")
    print("4. Load existing calibration")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            results = calibrator.calibrate_depth_accuracy()
            if results:
                calibrator.calibration_data.update(results)
                calibrator.save_calibration('depth_calibration.json')
        
        elif choice == '2':
            results = calibrator.calibrate_ground_plane()
            if results:
                calibrator.calibration_data.update(results)
                calibrator.save_calibration('ground_plane_calibration.json')
        
        elif choice == '3':
            print("Starting full calibration...")
            depth_results = calibrator.calibrate_depth_accuracy()
            if depth_results:
                calibrator.calibration_data.update(depth_results)
            
            ground_results = calibrator.calibrate_ground_plane()
            if ground_results:
                calibrator.calibration_data.update(ground_results)
            
            calibrator.save_calibration('full_calibration.json')
        
        elif choice == '4':
            filename = input("Enter calibration filename: ").strip()
            calibrator.load_calibration(filename)
        
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    
    finally:
        camera.stop()
        print("Calibration complete")

if __name__ == "__main__":
    main()
