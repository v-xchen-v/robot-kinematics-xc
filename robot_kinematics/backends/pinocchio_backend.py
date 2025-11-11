from .base_kinematics_backend import BaseKinematicsBackend

from typing import List, Optional, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation
from ..frames.transforms import T_to_pose, pose_to_T, Pose
from ..urdf.inspector import SubchainURDFInspector
from ..core.types import IKResult

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False

try:
    import casadi
    from pinocchio import casadi as cpin
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


class PinocchioKinematicsBackend(BaseKinematicsBackend):
    """
    Kinematics backend using Pinocchio.

    Notes
    -----
    - FK: uses pinocchio.framesForwardKinematics
    - Jacobian: uses pinocchio.computeFrameJacobian
    - IK: uses iterative Newton-Raphson method with Jacobian inverse
    """
    
    def __init__(
        self,
        urdf_path: str,
        base_link: str,
        ee_link: str,
        joint_names: Optional[List[str]] = None,
        package_dirs: Optional[str] = None,
        joints_to_lock: Optional[List[str]] = None,
        name: str = "pinocchio",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        if not PINOCCHIO_AVAILABLE:
            raise ImportError(
                "Pinocchio is not installed. Please install it with: "
                "conda install pinocchio -c conda-forge"
            )
        
        # Load the URDF model using RobotWrapper
        if package_dirs is not None:
            robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        else:
            robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
        
        # Build reduced robot if joints_to_lock is provided
        if joints_to_lock is not None:
            reference_config = np.zeros(robot.model.nq)
            robot = robot.buildReducedRobot(
                list_of_joints_to_lock=joints_to_lock,
                reference_configuration=reference_config,
            )
        
        # Store attributes needed for base class methods
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        
        # Initialize URDF inspector for shared methods (provides list_links, list_joints)
        self._urdf_inspector = SubchainURDFInspector(urdf_path, base_link, ee_link)
        
        # Get joint_names from the base class methods if not provided
        if joint_names is None:
            joint_names = self.list_joints(movable_only=True)
        link_names = self.list_links()
        
        # Find the end-effector frame ID
        if not robot.model.existFrame(ee_link):
            raise ValueError(f"End-effector link '{ee_link}' not found in the model frames.")
        ee_frame_id = robot.model.getFrameId(ee_link)
        
        # Set instance attributes
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.joint_names = joint_names
        self.link_names = link_names
        self.ee_frame_id = ee_frame_id
        self.n_dof = len(joint_names)
        
        # Initialize CasADi-based IK solver (lazy initialization)
        self._ik_solver_initialized = False
        self._casadi_opti = None
        self._casadi_var_q = None
        self._casadi_param_tf = None
        
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _ensure_q_shape(self, q: np.ndarray) -> np.ndarray:
        """Ensure joint configuration has the correct shape for Pinocchio."""
        if q.shape == (self.n_dof,):
            # Extend to full model configuration if needed
            q_full = pin.neutral(self.model)
            # Map the provided joints to the model's joint configuration
            # Assuming the joints are in the same order as the model's movable joints
            q_full[:self.n_dof] = q
            return q_full
        elif q.shape == (self.model.nq,):
            return q
        else:
            raise ValueError(
                f"Expected q shape {(self.n_dof,)} or {(self.model.nq,)}, got {q.shape}"
            )
    
    def find_frame_id_by_name(self, name: str) -> Optional[int]:
        """
        Find frame ID by frame name.
        
        Args:
            name: str
                Name of the frame to find.
        
        Returns:
            Optional[int]
                Frame ID if found, None otherwise.
        """
        for frame_id, frame in enumerate(self.model.frames):
            if frame.name == name:
                return frame_id
        return None
    
    def _initialize_ik_solver(self, solver_options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the CasADi-based IK solver.
        
        Args:
            solver_options: Optional[Dict[str, Any]]
                Options for the IPOPT solver.
        """
        if not CASADI_AVAILABLE:
            raise ImportError(
                "CasADi is not installed. Please install it for IK functionality: "
                "pip install casadi"
            )
        
        # Create CasADi model
        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        
        # Create symbolic variables
        cq = casadi.SX.sym("q", self.model.nq, 1)
        cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        
        # Define error function: calculate the difference between target pose and current end-effector pose
        error = casadi.Function(
            "error",
            [cq, cTf],
            [
                cpin.log6(
                    cdata.oMf[self.ee_frame_id].inverse() * cpin.SE3(cTf)
                ).vector
            ],
        )
        
        # Define optimization problem
        opti = casadi.Opti()
        var_q = opti.variable(self.model.nq)
        param_tf = opti.parameter(4, 4)
        
        # Objective function: minimize pose error
        totalcost = casadi.sumsqr(error(var_q, param_tf))
        
        # Set joint limit constraints
        opti.subject_to(opti.bounded(
            self.model.lowerPositionLimit,
            var_q,
            self.model.upperPositionLimit
        ))
        
        # Set optimization objective
        opti.minimize(totalcost)
        
        # Configure solver options
        if solver_options is None:
            solver_options = {
                'ipopt': {
                    'print_level': 0,
                    'max_iter': 200,
                    'tol': 1e-4,
                    'constr_viol_tol': 1e-4,
                    'dual_inf_tol': 1e-4,
                    'compl_inf_tol': 1e-4,
                    # Acceptable solution mechanism
                    'acceptable_tol': 1e-3,
                    'acceptable_iter': 15,
                    'acceptable_constr_viol_tol': 1e-3,
                    'acceptable_dual_inf_tol': 1e-3,
                    'acceptable_compl_inf_tol': 1e-3,
                    # Numerical stability improvements
                    'mu_strategy': 'adaptive',
                    'linear_solver': 'mumps',
                    'hessian_approximation': 'limited-memory',
                    'check_derivatives_for_naninf': 'yes',
                    # Numerical tolerance
                    'bound_relax_factor': 1e-8,
                    'honor_original_bounds': 'yes',
                    'nlp_scaling_method': 'gradient-based',
                },
                'print_time': False
            }
        
        opti.solver("ipopt", solver_options)
        
        # Store solver components
        self._casadi_opti = opti
        self._casadi_var_q = var_q
        self._casadi_param_tf = param_tf
        self._ik_solver_initialized = True
    
    # -------------------------------------------------------------------------
    # FK
    # -------------------------------------------------------------------------
    def fk(self, q: np.ndarray, link_name: Optional[str] = None) -> Pose:
        if link_name is None:
            link_name = self.ee_link
        
        # Get the frame ID
        if not self.model.existFrame(link_name):
            raise ValueError(f"Link '{link_name}' not found in the model frames.")
        frame_id = self.model.getFrameId(link_name)
        
        # Ensure q has the correct shape
        q_full = self._ensure_q_shape(q)
        
        # Compute forward kinematics using framesForwardKinematics
        pin.framesForwardKinematics(self.model, self.data, q_full)
        
        # Get the frame placement (SE3 transform)
        T = self.data.oMf[frame_id]
        
        # Convert to 4x4 homogeneous transformation matrix
        T_matrix = T.homogeneous
        
        return T_to_pose(T_matrix)
    
    def fk_all_frames(self, q: np.ndarray) -> Dict[str, Pose]:
        # Ensure q has the correct shape
        q_full = self._ensure_q_shape(q)
        
        # Compute forward kinematics using framesForwardKinematics
        pin.framesForwardKinematics(self.model, self.data, q_full)
        
        frames: Dict[str, Pose] = {}
        
        # Iterate over all frames
        for i in range(self.model.nframes):
            frame_name = self.model.frames[i].name
            T = self.data.oMf[i]
            T_matrix = T.homogeneous
            frames[frame_name] = T_to_pose(T_matrix)
        
        return frames

        
    # -------------------------------------------------------------------------
    # IK
    # -------------------------------------------------------------------------
    def ik(
        self,
        target_pose: Pose,
        initial_joint_positions: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        **kwargs: Any
    ) -> IKResult:
        """
        Inverse kinematics using CasADi optimization with Pinocchio.
        
        This method uses a constrained optimization approach with the IPOPT solver
        to find joint configurations that achieve the desired end-effector pose.
        
        Args:
            target_pose: Pose
                Desired target pose for the end-effector.
            initial_joint_positions: Optional[np.ndarray], optional
                Initial guess for joint positions. If None, use neutral configuration.
            max_iterations: int, optional
                Maximum number of iterations (passed to solver options). Default is 1000.
            tolerance: float, optional
                Convergence tolerance for pose error (passed to solver options). Default is 1e-4.
            damping: float, optional
                Not used in CasADi optimization (kept for API compatibility). Default is 1e-6.
            **kwargs: Any
                Additional arguments:
                - solver_options: Dict with custom IPOPT solver options
        
        Returns:
            IKResult
                Result object containing success status, joint positions, and error metrics.
        """
        # Initialize IK solver if not already done
        if not self._ik_solver_initialized:
            solver_options = kwargs.get('solver_options', None)
            if solver_options is None and (max_iterations != 1000 or tolerance != 1e-4):
                # Create custom solver options based on provided parameters
                solver_options = {
                    'ipopt': {
                        'print_level': 0,
                        'max_iter': max_iterations,
                        'tol': tolerance,
                        'constr_viol_tol': tolerance,
                        'dual_inf_tol': tolerance,
                        'compl_inf_tol': tolerance,
                        # Acceptable solution mechanism
                        'acceptable_tol': tolerance * 10,
                        'acceptable_iter': 15,
                        'acceptable_constr_viol_tol': tolerance * 10,
                        'acceptable_dual_inf_tol': tolerance * 10,
                        'acceptable_compl_inf_tol': tolerance * 10,
                        # Numerical stability improvements
                        'mu_strategy': 'adaptive',
                        'linear_solver': 'mumps',
                        'hessian_approximation': 'limited-memory',
                        'check_derivatives_for_naninf': 'yes',
                        # Numerical tolerance
                        'bound_relax_factor': 1e-8,
                        'honor_original_bounds': 'yes',
                        'nlp_scaling_method': 'gradient-based',
                    },
                    'print_time': False
                }
            self._initialize_ik_solver(solver_options)
        
        # Set initial joint positions
        if initial_joint_positions is None:
            initial_joint_positions = pin.neutral(self.model)
        else:
            initial_joint_positions = self._ensure_q_shape(initial_joint_positions)
        
        # Convert target pose to 4x4 transformation matrix
        target_matrix = pose_to_T(target_pose)
        
        # Set optimization parameters
        self._casadi_opti.set_initial(self._casadi_var_q, initial_joint_positions)
        self._casadi_opti.set_value(self._casadi_param_tf, target_matrix)
        
        # Solve optimization problem
        solved = False
        solver_info = {}
        try:
            sol = self._casadi_opti.solve_limited()
            sol_q = self._casadi_opti.value(self._casadi_var_q)
            solved = True
            solver_info['solver_status'] = 'success'
        except Exception as e:
            # If solver fails, get the best solution found so far
            solver_info['solver_status'] = 'failed'
            solver_info['error_message'] = str(e)
            sol_q = self._casadi_opti.debug.value(self._casadi_var_q)
        
        # Compute the actual achieved end-effector pose
        pin.framesForwardKinematics(self.model, self.data, sol_q)
        achieved_matrix = self.data.oMf[self.ee_frame_id].homogeneous
        achieved_pose = T_to_pose(achieved_matrix)
        
        # Compute position error (Euclidean distance)
        pos_err = np.linalg.norm(achieved_pose.xyz - target_pose.xyz)
        
        # Compute orientation error (angle difference between quaternions)
        # Convert quaternions to rotation matrices and compute geodesic distance
        target_rot = Rotation.from_quat([target_pose.quat_wxyz[1], target_pose.quat_wxyz[2], 
                                          target_pose.quat_wxyz[3], target_pose.quat_wxyz[0]])
        achieved_rot = Rotation.from_quat([achieved_pose.quat_wxyz[1], achieved_pose.quat_wxyz[2], 
                                            achieved_pose.quat_wxyz[3], achieved_pose.quat_wxyz[0]])
        
        # Compute relative rotation and extract angle
        relative_rot = target_rot.inv() * achieved_rot
        ori_err = relative_rot.magnitude()
        
        # Store additional info
        solver_info['achieved_pose'] = achieved_pose
        solver_info['iterations'] = max_iterations  # CasADi doesn't expose actual iteration count easily
        
        return IKResult(
            success=solved,
            q=sol_q,
            pos_err=pos_err,
            ori_err=ori_err,
            info=solver_info
        )
