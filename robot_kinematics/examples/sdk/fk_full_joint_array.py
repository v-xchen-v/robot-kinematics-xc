# FK with full joint array (chain order)

from robot_kinematics.core.robot_kinematics import RobotKinematics
import numpy as np
from robot_kinematics.core.types import Pose, IKOptions

def main():
    urdf_path = "robot_kinematics/data/g1/G1_120s/urdf/G1_120s.urdf"
    
    # Change backend_name here: "urdfpy" for fk
    robot_kinematics = RobotKinematics(
        urdf_path=urdf_path,
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="urdfpy",
    )
    # ---------- Joint order: robot.joint_names ----------
    print(f"Joint order (URDF) from {robot_kinematics.base_link} -> {robot_kinematics.ee_link}:")
    for name in robot_kinematics.joint_names:
        print(f"  - {name}")

    # ---------- FK with full joint array ----------
    q_home = np.zeros(robot_kinematics.n_dofs)
    # print the dict joint names and their values
    print(robot_kinematics.q_array_to_dict(q_home))
    
    pose_home = robot_kinematics.fk(q_home)
    print("\nFK (full joint array) - home pose:")
    print("  position   :", pose_home.xyz)
    print("  orientation:", pose_home.quat_wxyz)
    
    
    # Change backend to "pinocchio" 
    robot_kinematics_with_pin = RobotKinematics(
        urdf_path=urdf_path,
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="pinocchio",
    )
    
    pose_home_pin = robot_kinematics_with_pin.fk(q_home)
    print("\nFK with Pinocchio backend (full joint array) - home pose:")
    print("  position   :", pose_home_pin.xyz)
    print("  orientation:", pose_home_pin.quat_wxyz)
    
main()
    
    