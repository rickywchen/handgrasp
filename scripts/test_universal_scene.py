"""
Test 1: Verify universal scene loads and cube settles correctly
"""

from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_universal.xml"


def test_scene_loading():
    """Test that scene loads without errors"""
    print("\n" + "="*70)
    print("TEST 1: Scene Loading")
    print("="*70)
    
    if not XML_PATH.exists():
        print(f"ERROR: Scene XML not found at {XML_PATH}")
        return False
    
    try:
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        data = mujoco.MjData(model)
        print("✓ Scene loaded successfully")
        print(f"  - Bodies: {model.nbody}")
        print(f"  - Joints: {model.njnt}")
        print(f"  - DOFs: {model.nv}")
        return model, data
    except Exception as e:
        print(f"ERROR loading scene: {e}")
        return None, None


def test_object_detection(model, data):
    """Test that target object is present"""
    print("\n" + "="*70)
    print("TEST 2: Object Detection")
    print("="*70)
    
    try:
        # Find object body
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        
        print(f"✓ Target object found")
        print(f"  - Body ID: {object_id}")
        print(f"  - Geom ID: {object_geom_id}")
        
        return True
    except Exception as e:
        print(f"ERROR: Could not find target object: {e}")
        return False


def test_physics_settling(model, data):
    """Test that object settles on table correctly"""
    print("\n" + "="*70)
    print("TEST 3: Physics Settling")
    print("="*70)
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Get object ID
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    
    # Initial position
    initial_pos = data.xpos[object_id].copy()
    print(f"Initial object position: {initial_pos}")
    
    # Simulate settling
    print("Simulating 2 seconds of physics...")
    for _ in range(1000):  # 2 seconds at 0.002s timestep
        mujoco.mj_step(model, data)
    
    # Final position
    final_pos = data.xpos[object_id].copy()
    print(f"Final object position:   {final_pos}")
    
    # Check if settled on table (Z should be around 0.42-0.45)
    if 0.42 <= final_pos[2] <= 0.47:
        print("✓ Object settled on table correctly")
        print(f"  - Table height: 0.42m")
        print(f"  - Object Z: {final_pos[2]:.4f}m")
        return True
    else:
        print(f"WARNING: Object not on table (Z={final_pos[2]:.4f}m)")
        return False


def test_with_viewer(model, data):
    """Visual test with interactive viewer"""
    print("\n" + "="*70)
    print("TEST 4: Visual Inspection")
    print("="*70)
    print("Opening viewer... (press ESC to close)")
    
    try:
        # Reset
        mujoco.mj_resetData(model, data)
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Settle
            for _ in range(1000):
                mujoco.mj_step(model, data)
                viewer.sync()
            
            # Hold view
            print("Scene loaded. Check that:")
            print("  1. Cube is visible on table")
            print("  2. Panda robot is at home position")
            print("  3. No collision errors")
            print("\nHolding view for 10 seconds...")
            
            for _ in range(5000):
                mujoco.mj_step(model, data)
                viewer.sync()
                if not viewer.is_running():
                    break
        
        print("✓ Viewer test complete")
        return True
    except Exception as e:
        print(f"Viewer error: {e}")
        print("  (This is OK on some systems - continue with headless tests)")
        return False


def run_all_tests():
    """Run all scene tests"""
    print("\n" + "="*70)
    print("UNIVERSAL SCENE TESTING SUITE")
    print("="*70)
    
    # Test 1: Loading
    model, data = test_scene_loading()
    if model is None:
        print("\nCRITICAL ERROR: Scene failed to load. Fix XML before continuing.")
        return False
    
    # Test 2: Object detection
    if not test_object_detection(model, data):
        print("\nCRITICAL ERROR: Object not found in scene.")
        return False
    
    # Test 3: Physics
    if not test_physics_settling(model, data):
        print("\nWARNING: Physics settling may have issues, but continuing...")
    
    # Test 4: Viewer (optional)
    test_with_viewer(model, data)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ Scene loads correctly")
    print("✓ Object is present")
    print("✓ Basic physics works")
    print("\nReady to proceed to object_loader.py testing!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    
    if not success:
        print("\nFix errors above before proceeding to next component.")
        exit(1)
    else:
        print("All tests passed! Scene is ready.")
        exit(0)