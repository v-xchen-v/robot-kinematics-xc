"""
Example demonstrating IK functionality with Pinocchio backend.

This example shows how to:
1. Load a robot model with the Pinocchio backend
2. Compute forward kinematics to get a target pose
3. Solve inverse kinematics to reach that target pose
4. Verify the solution
"""

import numpy as np
from robot_kinematics.backends.pinocchio_backend import PinocchioKinematicsBackend
from robot_kinematics.core.types import Pose


def main():
    # Example paths - adjust these to your robot URDF
    # For this example, we'll use the G1 robot from the workspace
    urdf_path = "/home/yichao/Documents/repos/robot-kinematics-xc/robots/g1/G1_120s/urdf/G1_120s.urdf"
    package_dirs = ["/home/yichao/Documents/repos/robot-kinematics-xc/robots/g1/G1_120s/urdf"] # where to find mesh
    
    # Define the robot configuration
    base_link = "base_link"  # Adjust based on your robot
    ee_link = "gripper_r_center_link"  # Adjust based on your robot
    
    # Joints to lock (optional) - lock the right arm and leg for simpler IK
    joints_to_lock = [
        # 锁定身体关节
        "idx01_body_joint1",
        "idx02_body_joint2",
        
        # 锁定头部关节(ignore since it's not in Chain)
        
        # 锁定左臂所有关节(ignore since it's not in Chain)
        # "idx21_arm_l_joint1",
        # "idx22_arm_l_joint2",
        # "idx23_arm_l_joint3",
        # "idx24_arm_l_joint4",
        # "idx25_arm_l_joint5",
        # "idx26_arm_l_joint6",
        # "idx27_arm_l_joint7",
        # 锁定夹爪关节
        
        # 锁定所有夹爪关节（左右手) (ignore since is not in chain)
    ]
    
    print("=" * 60)
    print("Pinocchio Backend IK Example")
    print("=" * 60)
    
    # Initialize the backend
    print("\n1. Initializing Pinocchio backend...")
    try:
        backend = PinocchioKinematicsBackend(
            urdf_path=urdf_path,
            base_link=base_link,
            ee_link=ee_link,
            package_dirs=package_dirs,
            joints_to_lock=joints_to_lock,
        )
        print(f"   ✓ Backend initialized successfully")
        print(f"   - Number of DOF: {backend.n_dofs}")
        print(f"   - Joint names: {backend.joint_names}")
    except Exception as e:
        print(f"   ✗ Failed to initialize backend: {e}")
        print("\n   Note: Adjust the urdf_path, base_link, and ee_link for your robot.")
        return
    
    # Step 2: Define initial joint configuration
    print("\n2. Setting up initial joint configuration...")
    initial_q = np.zeros(backend.n_dofs)
    # You can set specific initial angles here if needed
    # initial_q[0] = 0.1  # Example: first joint at 0.1 rad
    print(f"   - Initial joint positions: {initial_q}")
    
    # Step 3: Compute forward kinematics to get a target pose
    print("\n3. Computing forward kinematics to get target pose...")
    target_pose = backend.fk(initial_q)
    print(f"   - Target position: {target_pose.xyz}")
    print(f"   - Target orientation (quat): {target_pose.quat_wxyz}")
    
    # Modify the target pose slightly to test IK
    print("\n4. Modifying target pose for IK test...")
    # Move the end-effector 0.1m in the z-direction
    target_pose.xyz[2] += 0.1
    print(f"   - New target position: {target_pose.xyz}")
    
    # Step 4: Solve inverse kinematics
    print("\n5. Solving inverse kinematics...")
    try:
        # IK now returns IKResult with success, q, pos_err, ori_err, info
        ik_result = backend.ik(
            target_pose=target_pose,
            initial_joint_positions=initial_q,
            max_iterations=200,
            tolerance=1e-4
        )
        
        solution_q = ik_result.q
        success = ik_result.success
        achieved_pose = ik_result.info.get('achieved_pose')
        
        print(f"   - IK converged: {success}")
        print(f"   - Solution joint positions: {solution_q}")
        
        # Step 5: Verify the solution
        print("\n6. Verifying IK solution...")
        print(f"   - Target position:   {target_pose.xyz}")
        print(f"   - Achieved position: {achieved_pose.xyz}")
        
        position_error = np.linalg.norm(target_pose.xyz - achieved_pose.xyz)
        print(f"   - Position error: {position_error:.6f} m")
        
        # Calculate orientation error (quaternion distance)
        # q1 · q2 gives the cosine of half the angle between them
        dot_product = np.abs(np.dot(target_pose.quat_wxyz, achieved_pose.quat_wxyz))
        angle_error = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
        print(f"   - Orientation error: {np.degrees(angle_error):.6f} degrees")
        
        if position_error < 0.01 and angle_error < np.radians(5):
            print("\n   ✓ IK solution is accurate!")
        else:
            print("\n   ⚠ IK solution has some error (may need tuning)")
            
    except Exception as e:
        print(f"   ✗ IK failed: {e}")
        print("\n   Note: Make sure CasADi is installed: pip install casadi")
        return
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
