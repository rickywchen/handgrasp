from pathlib import Path
from pathlib import Path
import mujoco
import mujoco.viewer

def main():
    # Load the table scene you edited
    xml_path = (Path(__file__).resolve().parents[1] / "assets" / "scenes" / "panda_table.xml").resolve()
    print("Loading:", xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
