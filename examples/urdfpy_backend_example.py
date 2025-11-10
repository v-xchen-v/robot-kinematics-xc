"""
Example usage of URDFPyKinematicsBackend

This example demonstrates how to:
1. Load a robot from a URDF file
2. Compute forward kinematics (FK)
3. Compute Jacobian matrices
4. List links and joints in the kinematic chain
"""

import numpy as np
from robot_kinematics.backends.urdfpy_backend import URDFPyKinematicsBackend


def main():
    # Example 1: Create backend with automatic joint discovery
    urdf_path = "robots/g1/G1_120s/urdf/G1_120s.urdf"
    base_link = "base_link"  # Example base link
    ee_link = "gripper_r_center_link"  # Example end-effector
    
    # Option A: Let the backend automatically discover joints
    backend = URDFPyKinematicsBackend.from_urdf(
        urdf_path=urdf_path,
        base_link=base_link,
        ee_link=ee_link,
        joint_names=None  # Will auto-discover movable joints
    )
    
    print(f"Backend created: {backend.name}")
    print(f"Number of DOFs: {backend.n_dof}")
    
    # List all joints in the kinematic chain
    joints = backend.list_joints(movable_only=True)
    print(f"\nJoints ({len(joints)}):")
    for i, joint in enumerate(joints, 1):
        print(f"  {i}. {joint}")
    
    # List all links in the kinematic chain
    links = backend.list_links()
    print(f"\nLinks ({len(links)}):")
    for i, link in enumerate(links, 1):
        print(f"  {i}. {link}")
    
    # Example 2: Forward Kinematics
    # Create a sample joint configuration (zeros for simplicity)
    q = np.zeros(backend.n_dof)
    
    # Compute FK for the end-effector
    ee_pose = backend.fk(q)
    print(f"\nEnd-effector pose at zero configuration:")
    print(f"  Position: {ee_pose.xyz}")
    print(f"  Quaternion (xyzw): {ee_pose.quat_wxyz}")
    
    # Example 3: FK for a specific link
    if len(links) > 2:
        mid_link = links[len(links) // 2]
        mid_pose = backend.fk(q, link_name=mid_link)
        print(f"\nPose of link '{mid_link}':")
        print(f"  Position: {mid_pose.xyz}")
    
    # Example 4: FK for all frames
    all_poses = backend.fk_all_frames(q)
    print(f"\nComputed FK for {len(all_poses)} frames")
    
    # Example 5: Create backend with explicit joint names
    # (useful when you want to control joint order or use a subset)
    specific_joints = joints[:3] if len(joints) >= 3 else joints
    backend2 = URDFPyKinematicsBackend.from_urdf(
        urdf_path=urdf_path,
        base_link=base_link,
        ee_link=ee_link,
        joint_names=specific_joints  # Explicit subset
    )
    print(f"\n\nBackend with specific joints:")
    print(f"  DOFs: {backend2.n_dof}")
    print(f"  Joints: {backend2.joint_names}")
    
    # Use with non-zero configuration
    q2 = np.random.uniform(-0.5, 0.5, backend2.n_dof)
    pose2 = backend2.fk(q2)
    print(f"\nEnd-effector pose with random configuration:")
    print(f"  Position: {pose2.xyz}")


if __name__ == "__main__":
    main()
