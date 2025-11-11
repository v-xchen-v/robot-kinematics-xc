from .base_kinematics_backend import BaseKinematicsBackend

from ..compat import urdfpy_compat
from urdfpy import URDF

from typing import List, Optional, Dict, Any
import numpy as np
from ..frames.transforms import T_to_pose, pose_to_T, Pose
from ..urdf.inspector import SubchainURDFInspector
from ..core.types import IKResult

class URDFPyKinematicsBackend(BaseKinematicsBackend):
    """
    Kinematics backend using urdfpy.

    Notes
    -----
    - FK: uses URDF.link_fk
    - Jacobian: uses URDF.jacobian
    - IK: Not implemented here because urdfpy does not support IK.
    """
    def __init__(
        self,
        urdf_path: str,
        base_link: str,
        ee_link: str,
        joint_names: Optional[List[str]] = None,
        name: str = "urdfpy",
        **kwargs: Any
    ):
        # Don't call super().__init__() yet - we need to set up attributes first
        
        # Load robot from URDF
        robot = URDF.load(urdf_path)
        
        # Store attributes needed for base class methods
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        
        
        # robot = RobotKinematics(
        #     urdf_path="assets/urdf/ur5.urdf",
        #     base_link="base_link",
        #     ee_link="tool0",
        #     backend_name="urdfpy",
        #     extra={"with_visuals": True},   # forwarded to URDFPyBackend(...)
        # )
        # if there is extra in kwargs, and inside the dict there is "with_visuals", we can pass it to urdfpy
        self.with_visuals = kwargs.get("with_visuals", False)

        
        self.name = name
        
        # Initialize URDF inspector for shared methods
        self._urdf_inspector = SubchainURDFInspector(urdf_path, base_link, ee_link)
        
        # Get joint_names and link_names from the base class methods
        if joint_names is None:
            joint_names = self.list_joints(movable_only=True)
        link_names = self.list_links()
        
        # Set instance attributes
        self.robot = robot
        self.joint_names = joint_names
        self.link_names = link_names
        self.n_dofs = len(joint_names)
        
    # -------------------------------------------------------------------------
    # Helpers for URDFPy
    # -------------------------------------------------------------------------
    def _q_to_dict(self, q: np.ndarray) -> Dict[str, float]:
        assert q.shape == (self.n_dofs,), f"Expected q shape {(self.n_dofs,)}, got {q.shape}"
        return {name: float(val) for name, val in zip(self.joint_names, q)}
    
        # -------------------------------------------------------------------------
    # FK
    # -------------------------------------------------------------------------
    def fk(self, joint_positions: np.ndarray, target_link: Optional[str] = None) -> Pose:
        if target_link is None:
            target_link = self.ee_link

        q_dict = self._q_to_dict(joint_positions)
        # link_fk returns dict: Link -> 4x4 transform; can also index by link name
        T_map = self.robot.link_fk(cfg=q_dict, use_names=True)
        # In some urdfpy versions T_map keys are Link objects; we can search by name
        T_target = None
        for link, T in T_map.items():
            if link == target_link:
                T_target = T
                break

        if T_target is None:
            raise ValueError(f"Link {target_link} not found in FK result.")

        
        if self.with_visuals:
            # Visualize the robot in the given configuration
            self.robot.show(cfg=q_dict)
        
        return T_to_pose(T_target)
    
    def fk_all_frames(self, joint_positions: np.ndarray) -> Dict[str, Pose]:
        q_dict = self._q_to_dict(joint_positions)
        T_map = self.robot.link_fk(cfg=q_dict, use_names=True)
        frames: Dict[str, Pose] = {}
        for link, T in T_map.items():
            name = link
            if name is None:
                continue
            frames[name] = T_to_pose(T)
        return frames

    # # -------------------------------------------------------------------------
    # # Jacobian
    # # -------------------------------------------------------------------------
    # def jacobian(
    #     self,
    #     joint_positions: np.ndarray,
    #     target_link: Optional[str] = None,
    # ) -> np.ndarray:
    #     raise NotImplementedError(
    #         "URDFPy backend does not support Jacobian computation. "
    #         "Please use a different backend (e.g., Pinocchio) for Jacobian capabilities."
    #     )
        
    # -------------------------------------------------------------------------
    # IK (Not Supported)
    # -------------------------------------------------------------------------
    def ik(
        self,
        target: Pose,
        initial_joint_positions: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> IKResult:
        """
        Inverse kinematics is not supported by urdfpy backend.
        
        Raises:
            NotImplementedError: Always, as urdfpy does not provide IK capabilities.
        """
        raise NotImplementedError(
            "URDFPy backend does not support inverse kinematics (IK). "
            "Please use a different backend (e.g., Pinocchio or RelaxIK) for IK capabilities."
        )


    # Helper method
    def q_array_to_dict(self, q: np.ndarray) -> Dict[str, float]:
        """
        When we get IK result q array, can use this helper user-friendly method to
        Check the joint values with names.
        
        Convert joint angle array to dictionary mapping joint names to values.
        
        Args:
            q: Joint angle array of shape (n_dof,)
            
        Returns:
            Dictionary {joint_name: joint_value}
        """
        assert q.shape == (self.n_dofs,), f"Expected q shape {(self.n_dofs,)}, got {q.shape}"
        return {name: float(val) for name, val in zip(self.joint_names, q)}