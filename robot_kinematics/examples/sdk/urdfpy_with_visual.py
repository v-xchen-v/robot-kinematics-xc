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
        extra={
            "with_visuals": True
        }
    )

    # home pose
    q_home = np.zeros(robot_kinematics.n_dofs)
    pose_home = robot_kinematics.fk(q_home)


    # 
    q_cfg = {
        'idx61_arm_r_joint1': 0.4,
        'idx62_arm_r_joint2': -1.4,
        'idx63_arm_r_joint3': -0.2,
        'idx64_arm_r_joint4': 1.2,
        'idx65_arm_r_joint5': -2.9,
    }
    init_pose = robot_kinematics.fk(q_cfg)

main()