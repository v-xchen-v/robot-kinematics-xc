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
    
    # ---------- FK with full joint array ----------
    q_home = np.zeros(robot_kinematics.n_dofs)
    
    q_cfg = robot_kinematics.q_array_to_dict(q_home)
    # print the dict joint names and their values
    print(q_cfg)
    
    pose_home = robot_kinematics.fk(q_cfg)
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
    
    pose_home_pin = robot_kinematics_with_pin.fk(q_cfg)
    print("\nFK with Pinocchio backend (full joint array) - home pose:")
    print("  position   :", pose_home_pin.xyz)
    print("  orientation:", pose_home_pin.quat_wxyz)
    
    
    # print joint limit for reference
    print("\nJoint limits:")
    joint_limits = robot_kinematics._backend.list_joint_limits()
    for name, joint_type, lower, upper in joint_limits:
        print(f"  {name:<20} [{lower:.2f}, {upper:.2f}]")
    
    # Only specify some joints, others as 0
    q_cfg = {
        "idx01_body_joint1": 0.3,
        "idx02_body_joint2": 0.5,
    }
    pose_pin = robot_kinematics_with_pin.fk(q_cfg)

    print("\nFK with Pinocchio backend (partial joint configuration) - pose:")
    print("  position   :", pose_pin.xyz)
    print("  orientation:", pose_pin.quat_wxyz)

main()
    
    