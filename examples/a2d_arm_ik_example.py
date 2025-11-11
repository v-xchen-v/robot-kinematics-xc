"""
Advanced A2D Arm IK Example using Pinocchio Backend.

This example demonstrates how to create specialized IK solvers for specific robots
(like the A2D dual-arm robot) using the robot_kinematics framework.

The implementation follows the pattern from the reference code but uses the
modular robot_kinematics framework.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from robot_kinematics.backends.pinocchio_backend import PinocchioKinematicsBackend
from robot_kinematics.core.types import Pose


def matrix_to_quat_pose(matrix):
    """
    Convert 4x4 transformation matrix to pose representation [x, y, z, qx, qy, qz, qw]
    Uses scipy.spatial.transform for more reliable conversion.
    """
    rotation = Rotation.from_matrix(matrix[:3, :3])
    return np.concatenate([matrix[:3, 3], rotation.as_quat()])


def quat_pose_to_matrix(pose):
    """
    Convert pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix.
    Uses scipy.spatial.transform for more reliable conversion.
    """
    rotation = Rotation.from_quat(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


class A2D_Arm_IK:
    """
    A2D dual-arm robot inverse kinematics solver.
    
    This class demonstrates how to use the robot_kinematics framework
    to create specialized IK solvers for specific robots.
    """
    
    def __init__(self, urdf_file_path, urdf_package_dirs, arm_side="left", 
                 joints_to_lock=None, init_angles=None, target_frame_name=None,
                 body_joint_values=None):
        """
        A2D dual-arm robot inverse kinematics solver.
        
        Args:
            urdf_file_path: Path to A2D.urdf file
            urdf_package_dirs: URDF package directories
            arm_side: "left" or "right" - which arm to control
            joints_to_lock: List of joints to lock
            init_angles: Initial joint angles
            target_frame_name: Target frame name, default is gripper_center
            body_joint_values: Body joint values [joint_lift_body, joint_body_pitch], default [0.0, 0.0]
        """
        
        self.arm_side = arm_side
        self.body_joint_values = body_joint_values if body_joint_values is not None else [0.0, 0.0]
        
        # Determine default joints to lock based on arm side
        if joints_to_lock is None:
            joints_to_lock = self._get_default_locked_joints(arm_side)
        
        # Determine target frame
        if target_frame_name is None:
            target_frame_name = f"{arm_side}_arm_joint7"
        
        # Initialize Pinocchio backend
        self.backend = PinocchioKinematicsBackend(
            urdf_path=urdf_file_path,
            base_link="base_link",  # Adjust based on your robot
            ee_link=target_frame_name,
            package_dirs=urdf_package_dirs,
            joints_to_lock=joints_to_lock,
        )
        
        self.target_frame_name = target_frame_name
        self.init_angles = init_angles if init_angles is not None else np.zeros(self.backend.n_dofs)
        
        print(f"Initialized {arm_side.upper()} arm IK solver")
        print(f"  - DOF: {self.backend.n_dofs}")
        print(f"  - Target frame: {target_frame_name}")
        print(f"  - Locked joints: {len(joints_to_lock)}")
    
    def _get_default_locked_joints(self, arm_side):
        """Get default list of joints to lock based on arm side."""
        base_locked_joints = [
            # Lock body joints
            "joint_lift_body",
            "joint_body_pitch",
            # Lock head joints
            "joint_head_yaw",
            "joint_head_pitch",
        ]
        
        # Lock the opposite arm
        if arm_side == "left":
            opposite_arm = "right"
        else:
            opposite_arm = "left"
        
        # Add opposite arm joints
        for i in range(1, 8):
            base_locked_joints.append(f"{opposite_arm}_arm_joint{i}")
        
        # Add gripper joints
        base_locked_joints.extend([
            "left_gripper_joint1",
            "right_gripper_joint1",
        ])
        
        return base_locked_joints
    
    def ik(self, target_matrix, init_angles=None, body_joint_values=None):
        """
        Inverse kinematics solver.
        
        Args:
            target_matrix: 4x4 target transformation matrix
            init_angles: Initial joint angles (optional)
            body_joint_values: Body joint values [joint_lift_body, joint_body_pitch] (optional)
            
        Returns:
            tuple: (joint_angles, success, actual_end_effector_pose_matrix)
        """
        if body_joint_values is not None:
            self.body_joint_values = body_joint_values
        
        if init_angles is not None:
            self.init_angles = init_angles
        
        # Convert matrix to Pose
        target_pose_array = matrix_to_quat_pose(target_matrix)
        target_pose = Pose(
            position=target_pose_array[:3],
            quaternion=target_pose_array[3:]
        )
        
        # Solve IK using backend
        ik_result = self.backend.ik(
            target_pose=target_pose,
            initial_joint_positions=self.init_angles,
            max_iterations=200,
            tolerance=1e-4
        )
        
        solution_q = ik_result.q
        success = ik_result.success
        achieved_pose = ik_result.info.get('achieved_pose')
        
        # Update initial angles for next iteration
        if success:
            self.init_angles = solution_q
        
        # Convert achieved pose back to matrix
        achieved_pose_array = np.concatenate([achieved_pose.position, achieved_pose.quaternion])
        achieved_matrix = quat_pose_to_matrix(achieved_pose_array)
        
        return solution_q, success, achieved_matrix
    
    def fk(self, joint_angles, return_matrix=False):
        """
        Forward kinematics computation.
        
        Args:
            joint_angles: Joint angle array
            return_matrix: Whether to return transformation matrix (default False, returns pose)
            
        Returns:
            Transformation matrix or pose [x,y,z,qx,qy,qz,qw]
        """
        pose = self.backend.fk(joint_angles)
        
        if return_matrix:
            pose_array = np.concatenate([pose.position, pose.quaternion])
            return quat_pose_to_matrix(pose_array)
        else:
            return np.concatenate([pose.position, pose.quaternion])
    
    def get_joint_names(self):
        """Get joint names in the simplified model."""
        return self.backend.joint_names
    
    def print_model_info(self):
        """Print model information."""
        print(f"=== A2D {self.arm_side.upper()} Arm IK Model Info ===")
        print(f"DOF: {self.backend.n_dofs}")
        print(f"Target frame: {self.target_frame_name}")
        print(f"Joint names: {self.backend.joint_names}")
        print("=" * 40)


def main():
    """Example usage of A2D Arm IK solver."""
    
    # Example paths - adjust for your A2D robot
    urdf_path = "../robots/g1/A2D_120s/urdf/A2D.urdf"
    package_dirs = ["../robots/g1/A2D_120s"]
    
    print("=" * 60)
    print("A2D Arm IK Example")
    print("=" * 60)
    
    try:
        # Create left arm IK solver
        print("\n1. Creating LEFT arm IK solver...")
        left_arm_ik = A2D_Arm_IK(
            urdf_path, 
            package_dirs, 
            arm_side="left"
        )
        left_arm_ik.print_model_info()
        
        # Test forward kinematics
        print("\n2. Testing forward kinematics...")
        body_joints = [0.3, 0.1]
        joint_angles = [0.1, -0.5, 0.3, -1.0, 0.5, 1.0, 0.2]
        
        left_matrix = left_arm_ik.fk(joint_angles, return_matrix=True)
        print(f"Left arm end-effector pose matrix:\n{left_matrix}")
        
        # Test inverse kinematics
        print("\n3. Testing inverse kinematics...")
        ik_result = left_arm_ik.ik(left_matrix)
        
        solution_q = ik_result.q
        success = ik_result.success
        achieved_matrix = ik_result.info.get('achieved_pose')
        
        print(f"IK solved: {success}")
        print(f"Solution joint angles: {solution_q}")
        print(f"Achieved pose: {achieved_matrix}")
        
        # Calculate error
        position_error = np.linalg.norm(left_matrix[:3, 3] - achieved_matrix[:3, 3])
        print(f"\nPosition error: {position_error:.6f} m")
        
        if position_error < 0.01:
            print("✓ IK solution is accurate!")
        else:
            print("⚠ IK solution has some error")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Adjust urdf_path and package_dirs for your robot.")
        print("Make sure CasADi is installed: pip install casadi")
        return
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
