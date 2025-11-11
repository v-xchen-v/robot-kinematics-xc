"""
Example of using/not using active joints in IK with Pinocchio backend.
"""

from robot_kinematics.core.robot_kinematics import RobotKinematics
import numpy as np
from robot_kinematics.core.types import Pose, IKOptions

def main():
    active_joints = [
        "idx61_arm_r_joint1",
        "idx62_arm_r_joint2",
        "idx63_arm_r_joint3",
        "idx64_arm_r_joint4",
        "idx65_arm_r_joint5",
        "idx66_arm_r_joint6",
        "idx67_arm_r_joint7",
    ]


    kin_pin = RobotKinematics(
        urdf_path="robot_kinematics/data/g1/G1_120s/urdf/G1_120s.urdf",
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="pinocchio",
        active_joints=active_joints,
    )

    print("Active joints:")
    print(kin_pin.active_joint_names)
    print("Inactive joints:")
    inactive_joints = [name for name in kin_pin.joint_names if name not in kin_pin.active_joint_names]
    print(inactive_joints)


    robot_kin_urdfpy = RobotKinematics(
        urdf_path="robot_kinematics/data/g1/G1_120s/urdf/G1_120s.urdf",
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="urdfpy",
    )

    # generate a reachable pose
    q_active_default = {
        'idx61_arm_r_joint1': 0.4,
        'idx62_arm_r_joint2': -1.4,
        'idx63_arm_r_joint3': -0.2,
        'idx64_arm_r_joint4': 1.2,
        'idx65_arm_r_joint5': -2.9,
        'idx66_arm_r_joint6': 0.0,
        'idx67_arm_r_joint7': 0.0,
    }
    q_full_default = kin_pin.expand_q_full(q_active_default, np.zeros(kin_pin.n_dofs))

    target_pose = robot_kin_urdfpy.fk(q_active_default)

    q_seed = q_active_default.copy()
    

    ik_result = kin_pin.ik(target_pose, seed_q=q_seed)
    q_arr = ik_result.q
    q_dict = kin_pin.q_array_to_dict(q_arr)
    print("IK result with active joints only:")
    print(q_dict)

    # delta q to check whether only active joints are changed
    delta_q = q_arr - list(q_seed.values())
    print("Delta q (should be non-zero only for active joints):")
    print(delta_q)
    for name, delta in zip(kin_pin.active_joint_names, delta_q):
        print(f"  {name:<20}: {delta:.4f}")
        
    # Calculate FK to verify IK result
    pose_fk = robot_kin_urdfpy.fk(q_dict)
    # delta of target and fk pose
    pos_err = np.linalg.norm(pose_fk.xyz - target_pose.xyz)
    ori_err = np.abs(np.arccos(np.clip(np.dot(pose_fk.quat_wxyz, target_pose.quat_wxyz), -1.0, 1.0))) * 2.0
    # print position error in mm
    print(f"FK of IK result - position error: {pos_err*1000:.6f}")
    # print orientation error in degrees
    print(f"FK of IK result - orientation error: {np.degrees(ori_err):.6f}")


def main_without_active_joints_cfg():
    kin_pin = RobotKinematics(
        urdf_path="robot_kinematics/data/g1/G1_120s/urdf/G1_120s.urdf",
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="pinocchio",
    )

    robot_kin_urdfpy = RobotKinematics(
        urdf_path="robot_kinematics/data/g1/G1_120s/urdf/G1_120s.urdf",
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="urdfpy",
    )

    # generate a reachable pose
    q_active_default = {
        'idx61_arm_r_joint1': 0.4,
        'idx62_arm_r_joint2': -1.4,
        'idx63_arm_r_joint3': -0.2,
        'idx64_arm_r_joint4': 1.2,
        'idx65_arm_r_joint5': -2.9,
        'idx66_arm_r_joint6': 0.0,
        'idx67_arm_r_joint7': 0.0,
    }
    target_pose = robot_kin_urdfpy.fk(q_active_default)

    q_seed = np.zeros(kin_pin.n_dofs)

    ik_result = kin_pin.ik(target_pose, seed_q=q_seed)
    q_arr = ik_result.q
    q_dict = kin_pin.q_array_to_dict(q_arr)
    print("IK result without active joints config:")
    print(q_dict)

main()
main_without_active_joints_cfg()