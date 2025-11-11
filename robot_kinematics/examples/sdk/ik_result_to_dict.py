from robot_kinematics.core.robot_kinematics import RobotKinematics
import numpy as np
from robot_kinematics.core.types import Pose

robot_kin = RobotKinematics(
    urdf_path="robot_kinematics/data/g1/G1_120s/urdf/G1_120s.urdf",
    base_link="base_link",
    ee_link="gripper_r_center_link",
    backend="pinocchio",
)

target_pose = Pose(
    xyz=np.array([0.4, 0.1, 0.2]),
    quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
)
q_seed = np.zeros(robot_kin.n_dofs)
ik_result = robot_kin.ik(target_pose, seed_q=q_seed)

q_dict = robot_kin.q_array_to_dict(ik_result.q)
print(q_dict)