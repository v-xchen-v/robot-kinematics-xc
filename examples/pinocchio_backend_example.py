"""
Example usage of the Pinocchio backend for robot kinematics.

This example demonstrates:
1. Loading a robot from URDF (with optional joint locking)
2. Computing forward kinematics (FK)
3. Computing the Jacobian matrix
4. Computing inverse kinematics (IK)
5. Computing FK for all frames
6. Finding frame IDs by name
"""

import numpy as np
from robot_kinematics.backends.pinocchio_backend import PinocchioKinematicsBackend
from robot_kinematics.frames import Pose

def main():
    # -------------------------------------------------------------------------
    # 1. Load the robot from URDF
    # -------------------------------------------------------------------------
    urdf_path = "robots/g1/G1_120s/urdf/G1_120s.urdf"
    base_link = "base_link"  # Example base link
    ee_link = "gripper_r_center_link"  # Example end-effector
    
    # Optional: List of joints to lock (similar to reference code)
    joints_to_lock = [
        # "left_Right_0_Joint",
        # "left_Right_1_Joint",
        # "left_Right_2_Joint",
        # "left_Right_Support_Joint",
        # "right_Right_0_Joint",
        # "right_Right_1_Joint",
        # "right_Right_2_Joint",
        # "right_Left_0_Joint",
        # "right_Left_2_Joint",
        # "right_Left_Support_Joint",
        # "right_Right_Support_Joint",
        # "left_Left_0_Joint",
        # "left_Left_2_Joint",
        # "left_Left_Support_Joint",
        # "left_hand_joint1",
        # "right_hand_joint1"
    ]
    
    print("Loading robot from URDF...")
    backend = PinocchioKinematicsBackend(
        urdf_path=urdf_path,
        base_link=base_link,
        ee_link=ee_link,
        joint_names=None,  # Will auto-detect movable joints
        package_dirs="robots/g1/G1_120s/urdf",  # Set to directory containing meshes if needed
        joints_to_lock=joints_to_lock if joints_to_lock else None,
    )
    
    print(f"Robot loaded successfully!")
    print(f"Number of DOF: {backend.n_dof}")
    print(f"Joint names: {backend.joint_names[:10]}...")  # Show first 10
    print(f"End-effector link: {backend.ee_link}")
    print()
    
    # -------------------------------------------------------------------------
    # 2. Forward Kinematics (FK)
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Forward Kinematics Example")
    print("=" * 70)
    
    # Create a random joint configuration
    q = np.random.uniform(-0.5, 0.5, backend.n_dof)
    print(f"Joint positions (first 10): {q[:10]}")
    
    # Compute FK for end-effector
    ee_pose = backend.fk(q)
    print(f"\nEnd-effector pose:")
    print(f"  Position: {ee_pose.xyz}")
    print(f"  Quaternion (xyzw): {ee_pose.quat_wxyz}")
    print()
    
    # -------------------------------------------------------------------------
    # 3. Find Frame ID by Name (similar to reference code)
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Find Frame ID Example")
    print("=" * 70)
    
    # Find frame IDs for specific links
    left_frame_id = backend.find_frame_id_by_name(ee_link)
    print(f"Frame ID for '{ee_link}': {left_frame_id}")
    
    # You can also access frames directly
    print(f"\nAll frames ({backend.model.nframes} total):")
    for frame_id in range(min(10, backend.model.nframes)):
        frame = backend.model.frames[frame_id]
        print(f"  Frame {frame_id}: {frame.name}")
    print()
    
    # # -------------------------------------------------------------------------
    # # 4. Jacobian Computation
    # # -------------------------------------------------------------------------
    # print("=" * 70)
    # print("Jacobian Computation Example")
    # print("=" * 70)
    
    # # Compute Jacobian at the current configuration
    # J = backend.jacobian(q)
    # print(f"Jacobian shape: {J.shape}")
    # print(f"Jacobian matrix (first 3 rows - linear velocity, first 5 cols):")
    # print(J[:3, :5])
    # print(f"\nJacobian matrix (last 3 rows - angular velocity, first 5 cols):")
    # print(J[3:, :5])
    # print()
    
    # -------------------------------------------------------------------------
    # 5. FK for All Frames
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("FK for All Frames Example")
    print("=" * 70)
    
    # Compute FK for all frames (similar to reference code)
    all_poses = backend.fk_all_frames(q)
    
    print(f"Total number of frames: {len(all_poses)}")
    print(f"\nSample frames and their positions:")
    for i, (frame_name, pose) in enumerate(list(all_poses.items())[:10]):
        print(f"  {frame_name:30s}: {pose.xyz}")
    if len(all_poses) > 10:
        print(f"  ... ({len(all_poses) - 10} more frames)")
    print()
    
    # -------------------------------------------------------------------------
    # 6. Inverse Kinematics (IK)
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Inverse Kinematics Example")
    print("=" * 70)
    
    # Define a target pose
    target_position = np.array([0.3, 0.2, 0.5])
    target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Identity rotation (xyzw)
    target_pose = Pose(xyz=target_position, quat_wxyz=target_quaternion)
    
    print(f"Target pose:")
    print(f"  Position: {target_pose.xyz}")
    print(f"  Quaternion: {target_pose.quat_wxyz}")
    
    # Compute IK
    print("\nComputing IK...")
    q_ik = backend.ik(
        target_pose=target_pose,
        initial_joint_positions=None,  # Start from neutral config
        max_iterations=1000,
        tolerance=1e-4,
        damping=1e-6,
    )
    
    print(f"IK solution (first 10): {q_ik[:10]}")
    
    # Verify the solution with FK
    ee_pose_ik = backend.fk(q_ik)
    print(f"\nVerification (FK of IK solution):")
    print(f"  Position: {ee_pose_ik.xyz}")
    print(f"  Quaternion: {ee_pose_ik.quat_wxyz}")
    
    # Compute position error
    position_error = np.linalg.norm(ee_pose_ik.xyz - target_pose.xyz)
    print(f"\nPosition error: {position_error:.6f} meters")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
