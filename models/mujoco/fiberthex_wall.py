import mujoco
import os

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
scene_path = os.path.join(base_dir, "scenes/wall.xml")

# Load scene with fiberthex and wall
FiberthexWallScene = mujoco.MjModel.from_xml_path(scene_path)