from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"

m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
d = mujoco.MjData(m)

for _ in range(10):
    mujoco.mj_step(m, d)

print("OK: loaded Panda")
print("nq =", m.nq, "nv =", m.nv, "nu =", m.nu)
