#!/usr/bin/env python3
"""
Test script to verify RealSense setup and dependencies
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'cv2',
        'numpy',
        'torch',
        'pyrealsense2',
        'ultralytics',
        'scipy',
        'filterpy',
        'matplotlib',
        'PIL'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Please install missing dependencies:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All imports successful!")
        return True

def test_realsense_camera():
    """Test RealSense camera connection"""
    print("\nTesting RealSense camera...")
    
    try:
        import pyrealsense2 as rs
        
        # Create pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get a frame
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if color_frame and depth_frame:
            print("✓ RealSense camera working!")
            print(f"  Color frame: {color_frame.get_width()}x{color_frame.get_height()}")
            print(f"  Depth frame: {depth_frame.get_width()}x{depth_frame.get_height()}")
            
            # Stop pipeline
            pipeline.stop()
            return True
        else:
            print("✗ No frames received from camera")
            pipeline.stop()
            return False
            
    except Exception as e:
        print(f"✗ RealSense camera error: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading"""
    print("\nTesting YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load a small model
        model = YOLO('yolov11n.pt')  # Nano model
        print("✓ YOLO model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ YOLO model error: {e}")
        return False

def test_custom_modules():
    """Test custom modules"""
    print("\nTesting custom modules...")
    
    try:
        from realsense_camera import RealSenseCamera
        from realsense_depth import RealSenseDepthEstimator
        from realsense_bbox3d_utils import RealSenseBBox3DEstimator
        print("✓ Custom modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Custom modules error: {e}")
        return False

def main():
    """Main test function"""
    print("=== RealSense YOLO-3D Setup Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("RealSense Camera Test", test_realsense_camera),
        ("YOLO Model Test", test_yolo_model),
        ("Custom Modules Test", test_custom_modules)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("You can now run: python run_realsense.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Connect RealSense camera and ensure it's not being used by another application")
        print("3. Check camera permissions and USB connection")
        print("4. Update camera firmware if needed")

if __name__ == "__main__":
    main()
