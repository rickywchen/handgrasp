import mujoco
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets" / "scenes" / "panda_table_mocap.xml"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Get body IDs
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

# Get world positions
hand_pos = data.xpos[hand_id]
left_pos = data.xpos[left_finger_id]
right_pos = data.xpos[right_finger_id]
cube_pos = data.xpos[cube_id]

print("="*60)
print("HAND GEOMETRY DIAGNOSIS")
print("="*60)
print(f"Hand body position:        {hand_pos}")
print(f"Left finger body position: {left_pos}")
print(f"Right finger body position: {right_pos}")
print(f"Cube position:             {cube_pos}")
print()
print(f"Hand to left finger offset:  {left_pos - hand_pos}")
print(f"Hand to right finger offset: {right_pos - hand_pos}")
print()
print(f"Distance hand → left finger:  {((left_pos - hand_pos)**2).sum()**0.5:.4f} m")
print(f"Distance hand → cube:         {((cube_pos - hand_pos)**2).sum()**0.5:.4f} m")
print()

# Check finger geoms
print("FINGER GEOMS:")
for i in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    
    if "finger" in body_name.lower():
        geom_type = model.geom_type[i]
        geom_size = model.geom_size[i]
        geom_pos = model.geom_pos[i]
        print(f"  {geom_name} (body: {body_name})")
        print(f"    Type: {geom_type}, Size: {geom_size}, Local pos: {geom_pos}")