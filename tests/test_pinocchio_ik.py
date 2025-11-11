"""
Unit tests for IK implementation in PinocchioKinematicsBackend.

These tests verify the inverse kinematics functionality.
"""

import numpy as np
import pytest
from robot_kinematics.backends.pinocchio_backend import PinocchioKinematicsBackend
from robot_kinematics.core.types import Pose


# Check if required dependencies are available
try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False

try:
    import casadi
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not installed")
@pytest.mark.skipif(not CASADI_AVAILABLE, reason="CasADi not installed")
class TestPinocchioIK:
    """Test suite for Pinocchio IK functionality."""
    
    @pytest.fixture
    def simple_backend(self):
        """Create a simple backend for testing (assumes G1 robot is available)."""
        try:
            backend = PinocchioKinematicsBackend(
                urdf_path="robots/g1/G1_120s/urdf/G1_120s.urdf",
                base_link="pelvis",
                ee_link="left_ankle_roll_link",
                package_dirs=["robots/g1/G1_120s"],
            )
            return backend
        except Exception:
            pytest.skip("Robot URDF not available for testing")
    
    def test_ik_basic_functionality(self, simple_backend):
        """Test that IK can be called without errors."""
        backend = simple_backend
        
        # Get a reachable target pose using FK
        initial_q = np.zeros(backend.n_dof)
        target_pose = backend.fk(initial_q)
        
        # Solve IK
        ik_result = backend.ik(target_pose=target_pose)
        
        # Check that solution has correct shape
        assert ik_result.q.shape == (backend.n_dof,) or ik_result.q.shape == (backend.model.nq,)
    
    def test_ik_with_return_success(self, simple_backend):
        """Test IK with IKResult return type."""
        backend = simple_backend
        
        # Get a reachable target pose
        initial_q = np.zeros(backend.n_dof)
        target_pose = backend.fk(initial_q)
        
        # Solve IK
        ik_result = backend.ik(target_pose=target_pose)
        
        # Check return types
        assert isinstance(ik_result.q, np.ndarray)
        assert isinstance(ik_result.success, bool)
        assert ik_result.info is not None
        achieved_pose = ik_result.info.get('achieved_pose')
        assert isinstance(achieved_pose, Pose)
        
        # Check that solution is valid
        assert not np.any(np.isnan(ik_result.q))
        assert not np.any(np.isinf(ik_result.q))
    
    def test_ik_accuracy(self, simple_backend):
        """Test that IK solution achieves the target pose within tolerance."""
        backend = simple_backend
        
        # Get a reachable target pose
        initial_q = np.zeros(backend.n_dof)
        target_pose = backend.fk(initial_q)
        
        # Solve IK
        ik_result = backend.ik(
            target_pose=target_pose,
            tolerance=1e-4
        )
        
        # Get achieved pose from result
        achieved_pose = ik_result.info.get('achieved_pose')
        
        # Calculate position error
        position_error = np.linalg.norm(target_pose.xyz - achieved_pose.xyz)
        
        # Check accuracy (allow some tolerance)
        assert position_error < 0.01, f"Position error too large: {position_error}"
    
    def test_ik_with_modified_target(self, simple_backend):
        """Test IK with a slightly modified target pose."""
        backend = simple_backend
        
        # Get initial pose
        initial_q = np.zeros(backend.n_dof)
        initial_pose = backend.fk(initial_q)
        
        # Modify target pose slightly (move in z direction)
        target_pose = Pose(
            xyz=initial_pose.xyz + np.array([0.0, 0.0, 0.05]),
            quat_wxyz=initial_pose.quat_wxyz
        )
        
        # Solve IK
        ik_result = backend.ik(
            target_pose=target_pose,
            initial_joint_positions=initial_q,
        )
        
        solution = ik_result.q
        success = ik_result.success
        achieved_pose = ik_result.info.get('achieved_pose')
        
        # Check that solution is different from initial
        assert not np.allclose(solution[:backend.n_dof], initial_q)
        
        # Check achieved pose is close to target
        position_error = np.linalg.norm(target_pose.xyz - achieved_pose.xyz)
        assert position_error < 0.1, f"Position error: {position_error}"
    
    def test_ik_respects_joint_limits(self, simple_backend):
        """Test that IK solution respects joint limits."""
        backend = simple_backend
        
        # Get a reachable target
        initial_q = np.zeros(backend.n_dof)
        target_pose = backend.fk(initial_q)
        
        # Solve IK
        ik_result = backend.ik(target_pose=target_pose)
        
        solution = ik_result.q
        
        # Ensure solution fits model size
        if solution.shape[0] > backend.n_dof:
            solution = solution[:backend.n_dof]
        
        # Check joint limits (with small tolerance for numerical errors)
        lower_limits = backend.model.lowerPositionLimit[:backend.n_dof]
        upper_limits = backend.model.upperPositionLimit[:backend.n_dof]
        
        tolerance = 1e-6
        assert np.all(solution >= lower_limits - tolerance), \
            f"Solution violates lower limits: {solution} < {lower_limits}"
        assert np.all(solution <= upper_limits + tolerance), \
            f"Solution violates upper limits: {solution} > {upper_limits}"
    
    def test_ik_with_initial_guess(self, simple_backend):
        """Test IK with different initial guesses."""
        backend = simple_backend
        
        # Get target pose
        target_q = np.zeros(backend.n_dof)
        target_pose = backend.fk(target_q)
        
        # Solve IK with different initial guesses
        initial_guess_1 = np.zeros(backend.n_dof)
        initial_guess_2 = np.ones(backend.n_dof) * 0.1
        
        ik_result_1 = backend.ik(
            target_pose=target_pose,
            initial_joint_positions=initial_guess_1
        )
        ik_result_2 = backend.ik(
            target_pose=target_pose,
            initial_joint_positions=initial_guess_2
        )
        
        solution_1 = ik_result_1.q
        solution_2 = ik_result_2.q
        
        # Both should be valid solutions
        assert not np.any(np.isnan(solution_1))
        assert not np.any(np.isnan(solution_2))
        
        # Verify both achieve similar end-effector poses
        pose_1 = backend.fk(solution_1[:backend.n_dof])
        pose_2 = backend.fk(solution_2[:backend.n_dof])
        
        error_1 = np.linalg.norm(target_pose.xyz - pose_1.xyz)
        error_2 = np.linalg.norm(target_pose.xyz - pose_2.xyz)
        
        assert error_1 < 0.01
        assert error_2 < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
