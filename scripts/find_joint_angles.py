"""
Interactive tool to find correct joint angles for grasping.
Use the viewer GUI to manually position the arm, then save the joint angles.
"""

from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    
    # Initialize
    mujoco.mj_resetData(model, data)
    
    # Set initial pose
    data.qpos[:7] = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785]
    data.qpos[7:9] = 0.04
    
    # Place cube on table
    data.qpos[9:12] = [0.6, 0.15, 0.445]
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    
    mujoco.mj_forward(model, data)
    
    # Get IDs
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    
    print("\n" + "="*60)
    print("INTERACTIVE JOINT ANGLE FINDER")
    print("="*60)
    print("\nInstructions:")
    print("1. Use viewer GUI: right-click and drag to move joints")
    print("2. Double-click a body to select it")
    print("3. Use Ctrl+Right-drag to perturb")
    print("4. Position hand above/at cube")
    print("5. Press Ctrl+P in viewer to print current state")
    print("\nCube position: [0.6, 0.15, 0.445]")
    print("\nYou need 3 poses:")
    print("  - Approach: hand above cube (~20cm)")
    print("  - Grasp: hand at cube height")
    print("  - Lift: hand raised up")
    print("\nPress Enter when in a good pose, then copy the joint angles below.")
    print("="*60 + "\n")
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    pose_count = 0
    pose_names = ["APPROACH", "GRASP", "LIFT"]
    
    try:
        while viewer.is_running():
            # Update display
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Print current state every 100 steps
            if data.time % 0.5 < 0.002:  # Every 0.5 seconds
                hand_pos = data.xpos[hand_body_id]
                cube_pos = data.xpos[cube_body_id]
                dist = np.linalg.norm(hand_pos - cube_pos)
                
                joint_angles = data.qpos[:7]
                
                print(f"\rHand: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]  "
                      f"Dist to cube: {dist:.3f}m  "
                      f"Joints: {np.array2string(joint_angles, precision=3, suppress_small=True)}", 
                      end='', flush=True)
            
            # Check for user input (simple - they'll copy-paste the values)
            
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    
    print("\n\nTo use these angles:")
    print("1. Position arm in desired pose")
    print("2. Copy the 'Joints' values from above")
    print("3. Paste into pickup_clean.py as q_approach, q_grasp, or q_lift")
    print("\nExample:")
    print("  q_approach = np.array([0.3, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785])")


if __name__ == "__main__":
    main()