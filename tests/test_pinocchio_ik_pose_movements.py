"""
Pinocchio IK with specific target pose and various movements.

This script uses a given base target pose and applies:
- 5cm translations in x, y, z directions
- 5 degree rotations around each axis
Then verifies IK accuracy for each movement.
"""

import numpy as np
from robot_kinematics.backends.pinocchio_backend import PinocchioKinematicsBackend
from robot_kinematics.frames.transforms import Pose
from scipy.spatial.transform import Rotation


# Check if required dependencies are available
try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    print("Warning: Pinocchio not installed")

try:
    import casadi
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    print("Warning: CasADi not installed")


def transformation_matrix_to_pose(T):
    """Convert 4x4 transformation matrix to Pose object."""
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
    quaternion_wxyz = np.array([quaternion_xyzw[3], quaternion_xyzw[0], 
                                 quaternion_xyzw[1], quaternion_xyzw[2]])
    return Pose(xyz=position, quat_wxyz=quaternion_wxyz)


def pose_to_transformation_matrix(pose):
    """Convert Pose object to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, 3] = pose.position
    # Convert quaternion wxyz to rotation matrix
    quat_xyzw = np.array([pose.quaternion[1], pose.quaternion[2], 
                          pose.quaternion[3], pose.quaternion[0]])
    rotation = Rotation.from_quat(quat_xyzw)
    T[:3, :3] = rotation.as_matrix()
    return T


def apply_translation(pose, dx=0, dy=0, dz=0):
    """Apply translation to a pose."""
    new_position = pose.xyz + np.array([dx, dy, dz])
    return Pose(xyz=new_position, quat_wxyz=pose.quat_wxyz)


def apply_rotation(pose, roll=0, pitch=0, yaw=0):
    """Apply rotation (in degrees) to a pose.
    
    Args:
        pose: Input pose
        roll: Rotation around x-axis in degrees
        pitch: Rotation around y-axis in degrees  
        yaw: Rotation around z-axis in degrees
    """
    # Convert current quaternion to rotation matrix
    quat_xyzw = np.array([pose.quat_wxyz[1], pose.quat_wxyz[2], 
                          pose.quat_wxyz[3], pose.quat_wxyz[0]])
    current_rot = Rotation.from_quat(quat_xyzw)
    
    # Create additional rotation
    additional_rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    
    # Combine rotations
    new_rot = current_rot * additional_rot
    
    # Convert back to quaternion wxyz
    new_quat_xyzw = new_rot.as_quat()
    new_quat_wxyz = np.array([new_quat_xyzw[3], new_quat_xyzw[0], 
                               new_quat_xyzw[1], new_quat_xyzw[2]])
    
    return Pose(xyz=pose.xyz.copy(), quat_wxyz=new_quat_wxyz)


def calculate_pose_error(target_pose, achieved_pose):
    """Calculate position and orientation errors between two poses.
    
    Returns:
        position_error: L2 norm of position difference in meters
        orientation_error: Angular difference in degrees
    """
    # Position error
    position_error = np.linalg.norm(target_pose.xyz - achieved_pose.xyz)
    
    # Orientation error (quaternion distance)
    dot_product = np.abs(np.dot(target_pose.quat_wxyz, achieved_pose.quat_wxyz))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_error = 2 * np.arccos(dot_product)
    orientation_error_deg = np.degrees(angle_error)
    
    return position_error, orientation_error_deg


# Base target pose from user (4x4 transformation matrix)
BASE_TARGET_MATRIX = np.array([
    [ 0.981565,  0.049909,  0.184498,  0.588399],
    [-0.178072, -0.111866,  0.977638, -0.301971],
    [ 0.069432, -0.992469, -0.100916,  1.214408],
    [ 0.0,       0.0,       0.0,       1.0      ]
])


def create_backend():
    """Create backend for testing (G1 robot left arm)."""
    try:
        backend = PinocchioKinematicsBackend(
            urdf_path="robots/g1/G1_120s/urdf/G1_120s.urdf",
            base_link="base_link",
            ee_link="gripper_r_center_link",
            package_dirs=["robots/g1/G1_120s/urdf"],
        )
        return backend
    except Exception as e:
        print(f"Error: Robot URDF not available: {e}")
        return None


def get_base_target_pose():
    """Convert base target matrix to Pose object."""
    return transformation_matrix_to_pose(BASE_TARGET_MATRIX)


def test_base_target_pose_ik(backend, base_target_pose):
    """Test IK on the base target pose."""
    print("\n" + "="*70)
    print("Testing Base Target Pose")
    print("="*70)
    print(f"Target position: {base_target_pose.xyz}")
    print(f"Target quaternion (wxyz): {base_target_pose.quat_wxyz}")
    
    # Get initial joint positions (zero configuration)
    initial_q = np.zeros(backend.n_dof)
    
    # Solve IK
    solution_q, success, achieved_pose = backend.ik(
        target_pose=base_target_pose,
        initial_joint_positions=initial_q,
        return_success=True,
        max_iterations=500,
        tolerance=1e-4
    )
    
    # Calculate errors
    pos_error, ori_error = calculate_pose_error(base_target_pose, achieved_pose)
    
    print(f"\nIK Results:")
    print(f"  Converged: {success}")
    print(f"  Solution: {solution_q}")
    print(f"  Position error: {pos_error*1000:.2f} mm")
    print(f"  Orientation error: {ori_error:.2f} degrees")
    
    # Validations
    if not success:
        print("  WARNING: IK did not converge for base target pose")
    if pos_error >= 0.02:
        print(f"  WARNING: Position error too large: {pos_error*1000:.2f} mm")
    if ori_error >= 10.0:
        print(f"  WARNING: Orientation error too large: {ori_error:.2f} degrees")
    
    return success


def test_translation_5cm(backend, base_target_pose, direction, dx, dy, dz):
    """Test IK accuracy with 5cm translation in each direction."""
    print("\n" + "="*70)
    print(f"Testing 5cm Translation in {direction.upper()} direction")
    print("="*70)
    
    # Apply translation
    target_pose = apply_translation(base_target_pose, dx, dy, dz)
    print(f"Original position: {base_target_pose.xyz}")
    print(f"Target position:   {target_pose.xyz}")
    print(f"Translation: [{dx*1000:.0f}, {dy*1000:.0f}, {dz*1000:.0f}] mm")
    
    # Get initial joint positions
    initial_q = np.zeros(backend.n_dof)
    
    # Solve IK
    solution_q, success, achieved_pose = backend.ik(
        target_pose=target_pose,
        initial_joint_positions=initial_q,
        return_success=True,
        max_iterations=500,
        tolerance=1e-4
    )
    
    # Calculate errors
    pos_error, ori_error = calculate_pose_error(target_pose, achieved_pose)
    
    print(f"\nIK Results:")
    print(f"  Converged: {success}")
    print(f"  Achieved position: {achieved_pose.xyz}")
    print(f"  Position error: {pos_error*1000:.2f} mm")
    print(f"  Orientation error: {ori_error:.2f} degrees")
    
    # Validations (allow larger tolerance for translation)
    if success:
        if pos_error >= 0.05:
            print(f"  WARNING: Position error too large: {pos_error*1000:.2f} mm")
        if ori_error >= 15.0:
            print(f"  WARNING: Orientation error too large: {ori_error:.2f} degrees")
    else:
        print(f"  INFO: IK did not converge for {direction} translation (may be unreachable)")
    
    return success


def test_rotation_5deg(backend, base_target_pose, axis, roll, pitch, yaw):
    """Test IK accuracy with 5 degree rotation around each axis."""
    print("\n" + "="*70)
    print(f"Testing 5° Rotation around {axis.upper()} axis")
    print("="*70)
    
    # Apply rotation
    target_pose = apply_rotation(base_target_pose, roll, pitch, yaw)
    print(f"Original quaternion: {base_target_pose.quat_wxyz}")
    print(f"Target quaternion:   {target_pose.quat_wxyz}")
    print(f"Rotation: roll={roll}°, pitch={pitch}°, yaw={yaw}°")
    
    # Get initial joint positions
    initial_q = np.zeros(backend.n_dof)
    
    # Solve IK
    solution_q, success, achieved_pose = backend.ik(
        target_pose=target_pose,
        initial_joint_positions=initial_q,
        return_success=True,
        max_iterations=500,
        tolerance=1e-4
    )
    
    # Calculate errors
    pos_error, ori_error = calculate_pose_error(target_pose, achieved_pose)
    
    print(f"\nIK Results:")
    print(f"  Converged: {success}")
    print(f"  Achieved quaternion: {achieved_pose.quat_wxyz}")
    print(f"  Position error: {pos_error*1000:.2f} mm")
    print(f"  Orientation error: {ori_error:.2f} degrees")
    
    # Validations
    if success:
        if pos_error >= 0.05:
            print(f"  WARNING: Position error too large: {pos_error*1000:.2f} mm")
        if ori_error >= 10.0:
            print(f"  WARNING: Orientation error too large: {ori_error:.2f} degrees")
    else:
        print(f"  INFO: IK did not converge for {axis} rotation (may be unreachable)")
    
    return success


def test_combined_movement(backend, base_target_pose):
    """Test IK with combined translation (5cm diagonal) and rotation (5deg)."""
    print("\n" + "="*70)
    print("Testing Combined Movement (5cm + 5° rotation)")
    print("="*70)
    
    # Apply translation (5cm in each direction - diagonal movement)
    target_pose = apply_translation(base_target_pose, 0.05, 0.05, 0.05)
    
    # Apply rotation (5 degrees around z-axis)
    target_pose = apply_rotation(target_pose, 0.0, 0.0, 5.0)
    
    print(f"Original position: {base_target_pose.xyz}")
    print(f"Target position:   {target_pose.xyz}")
    print(f"Translation: [50, 50, 50] mm")
    print(f"Rotation: 5° around z-axis")
    
    # Get initial joint positions
    initial_q = np.zeros(backend.n_dof)
    
    # Solve IK
    solution_q, success, achieved_pose = backend.ik(
        target_pose=target_pose,
        initial_joint_positions=initial_q,
        return_success=True,
        max_iterations=500,
        tolerance=1e-4
    )
    
    # Calculate errors
    pos_error, ori_error = calculate_pose_error(target_pose, achieved_pose)
    
    print(f"\nIK Results:")
    print(f"  Converged: {success}")
    print(f"  Position error: {pos_error*1000:.2f} mm")
    print(f"  Orientation error: {ori_error:.2f} degrees")
    
    # Validations (allow larger tolerance for combined movement)
    if success:
        if pos_error >= 0.1:
            print(f"  WARNING: Position error too large: {pos_error*1000:.2f} mm")
        if ori_error >= 20.0:
            print(f"  WARNING: Orientation error too large: {ori_error:.2f} degrees")
    else:
        print("  INFO: IK did not converge for combined movement (may be unreachable)")
    
    return success


def test_accuracy_summary(backend, base_target_pose):
    """Generate a comprehensive accuracy summary for all movements."""
    print("\n" + "="*70)
    print("IK ACCURACY SUMMARY")
    print("="*70)
    
    movements = [
        ("Base", base_target_pose),
        ("+5cm X", apply_translation(base_target_pose, 0.05, 0, 0)),
        ("+5cm Y", apply_translation(base_target_pose, 0, 0.05, 0)),
        ("+5cm Z", apply_translation(base_target_pose, 0, 0, 0.05)),
        ("+5° Roll", apply_rotation(base_target_pose, 5, 0, 0)),
        ("+5° Pitch", apply_rotation(base_target_pose, 0, 5, 0)),
        ("+5° Yaw", apply_rotation(base_target_pose, 0, 0, 5)),
    ]
    
    print(f"\n{'Movement':<15} {'Converged':<12} {'Pos Error (mm)':<18} {'Ori Error (°)':<15}")
    print("-" * 70)
    
    results = []
    initial_q = np.zeros(backend.n_dof)
    
    for name, target_pose in movements:
        try:
            solution_q, success, achieved_pose = backend.ik(
                target_pose=target_pose,
                initial_joint_positions=initial_q,
                return_success=True,
                max_iterations=500,
                tolerance=1e-4
            )
            
            pos_error, ori_error = calculate_pose_error(target_pose, achieved_pose)
            results.append((name, success, pos_error, ori_error))
            
            status = "✓" if success else "✗"
            print(f"{name:<15} {status:<12} {pos_error*1000:>15.2f}   {ori_error:>13.2f}")
            
        except Exception as e:
            print(f"{name:<15} ERROR: {str(e)}")
            results.append((name, False, float('inf'), float('inf')))
    
    print("="*70)
    
    # Calculate statistics for successful cases
    successful_results = [(p, o) for _, s, p, o in results if s and p < float('inf')]
    if successful_results:
        pos_errors, ori_errors = zip(*successful_results)
        print(f"\nStatistics (successful cases):")
        print(f"  Position error - Mean: {np.mean(pos_errors)*1000:.2f} mm, "
              f"Max: {np.max(pos_errors)*1000:.2f} mm")
        print(f"  Orientation error - Mean: {np.mean(ori_errors):.2f}°, "
              f"Max: {np.max(ori_errors):.2f}°")
    
    # At least some movements should succeed
    success_count = sum(1 for _, s, _, _ in results if s)
    print(f"\nSuccess rate: {success_count}/{len(movements)}")
    
    if success_count == 0:
        print("WARNING: No IK solutions succeeded!")
    
    return results


def main():
    """Main function to run all tests."""
    if not PINOCCHIO_AVAILABLE or not CASADI_AVAILABLE:
        print("Error: Required dependencies not available")
        print(f"  Pinocchio: {PINOCCHIO_AVAILABLE}")
        print(f"  CasADi: {CASADI_AVAILABLE}")
        return
    
    # Create backend
    backend = create_backend()
    if backend is None:
        print("Error: Failed to create backend")
        return
    
    # Get base target pose
    base_target_pose = get_base_target_pose()
    
    # Run tests
    print("\n" + "="*70)
    print("STARTING IK POSE MOVEMENT TESTS")
    print("="*70)
    
    # Test 1: Base target pose
    test_base_target_pose_ik(backend, base_target_pose)
    
    # Test 2: Translations
    translations = [
        ("x", 0.05, 0.0, 0.0),
        ("y", 0.0, 0.05, 0.0),
        ("z", 0.0, 0.0, 0.05),
    ]
    for direction, dx, dy, dz in translations:
        test_translation_5cm(backend, base_target_pose, direction, dx, dy, dz)
    
    # Test 3: Rotations
    rotations = [
        ("x", 5.0, 0.0, 0.0),
        ("y", 0.0, 5.0, 0.0),
        ("z", 0.0, 0.0, 5.0),
    ]
    for axis, roll, pitch, yaw in rotations:
        test_rotation_5deg(backend, base_target_pose, axis, roll, pitch, yaw)
    
    # Test 4: Combined movement
    test_combined_movement(backend, base_target_pose)
    
    # Test 5: Accuracy summary
    test_accuracy_summary(backend, base_target_pose)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
