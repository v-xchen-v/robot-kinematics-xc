"""
Example of using/not using active joints in IK with Pinocchio backend.
"""

from robot_kinematics.core.robot_kinematics import RobotKinematics
import numpy as np
from robot_kinematics.core.types import Pose, IKOptions

def main():
    robot_kin = RobotKinematics(
        urdf_path="robot_kinematics/data/g1/G1_120s/urdf/G1_120s.urdf",
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="pinocchio",
        active_joints=[
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
        ]
    )
    
    target_pose = Pose(
        xyz=np.array([0.4, 0.1, 0.2]),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    q_seed = np.zeros(robot_kin.n_dofs)
    ik_result = robot_kin.ik(target_pose, seed_q=q_seed)    
    q_arr = ik_result.q
    q_dict = robot_kin.q_array_to_dict(q_arr)
    print("IK result with active joints only:")
    print(q_dict)
    
    # delta q to check whether only active joints are changed
    delta_q = q_arr - q_seed[robot_kin.active_joint_indices]
    print("Delta q (should be non-zero only for active joints):")
    print(delta_q)
    for name, delta in zip(robot_kin.active_joint_names, delta_q):
        print(f"  {name:<20}: {delta:.4f}")
        
    # # update the full q with inactive qseed + q_arr in ik_result do fk and visualize to check
    # q_full = q_seed.copy()
    # for i, name in enumerate(robot_kin.active_joint_names):
    #     joint_index = robot_kin._name_to_index[name]
    #     q_full[joint_index] = q_arr[i]
    pose_fk = robot_kin.fk(q_arr)
    print("\nFK of IK result - pose:")
    print("  position   :", pose_fk.xyz)
    print("  orientation:", pose_fk.quat_wxyz)
    
main()