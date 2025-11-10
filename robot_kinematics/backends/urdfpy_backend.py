from .base_kinematics_backend import BaseKinematicsBackend

from ..compat import urdfpy_compat
from urdfpy import URDF

from typing import List, Optional, Dict, Any
import numpy as np
from ..frames.transforms import T_to_pose, pose_to_T, Pose

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
        robot: URDF,
        base_link: str,
        ee_link: str,
        joint_names: List[str],
        name: str = "urdfpy",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, metadata=metadata)
        self.robot = robot
        self.base_link = base_link
        self.ee_link = ee_link
        self.joint_names = joint_names
        self.n_dof = len(joint_names)
        
    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        base_link: str,
        ee_link: str,
        joint_names: Optional[List[str]] = None,
        **kwargs: Any
    ) -> "URDFPyKinematicsBackend":
        robot = URDF.load(urdf_path)

        # Create a temporary instance to use base class methods
        temp_instance = cls.__new__(cls)
        temp_instance.urdf_path = urdf_path
        temp_instance.base_link = base_link
        temp_instance.ee_link = ee_link
        temp_instance._urdf_inspector = None
        
        # Get joint_names and link_names from the base class methods
        if joint_names is None:
            joint_names = temp_instance.list_joints(movable_only=True)
        link_names = temp_instance.list_links()
            
        metadata = {
            "urdf_path": urdf_path,
            "base_link": base_link,
            "ee_link": ee_link,
        }
        metadata.update(kwargs)

        instance = cls(
            robot=robot,
            base_link=base_link,
            ee_link=ee_link,
            joint_names=joint_names,
            metadata=metadata,
        )
        # Store urdf_path so base class methods can use it
        instance.urdf_path = urdf_path
        instance.joint_names = joint_names
        instance.link_names = link_names
        return instance
        
    # -------------------------------------------------------------------------
    # Helpers for URDFPy
    # -------------------------------------------------------------------------
    def _q_to_dict(self, q: np.ndarray) -> Dict[str, float]:
        assert q.shape == (self.n_dof,), f"Expected q shape {(self.n_dof,)}, got {q.shape}"
        return {name: float(val) for name, val in zip(self.joint_names, q)}
    
        # -------------------------------------------------------------------------
    # FK
    # -------------------------------------------------------------------------
    def fk(self, q: np.ndarray, link_name: Optional[str] = None) -> Pose:
        if link_name is None:
            link_name = self.ee_link

        q_dict = self._q_to_dict(q)
        # link_fk returns dict: Link -> 4x4 transform; can also index by link name
        T_map = self.robot.link_fk(cfg=q_dict, use_names=True)
        # In some urdfpy versions T_map keys are Link objects; we can search by name
        T_target = None
        for link, T in T_map.items():
            if link == link_name:
                T_target = T
                break

        if T_target is None:
            raise ValueError(f"Link {link_name} not found in FK result.")

        return T_to_pose(T_target)
    
    def fk_all_frames(self, q: np.ndarray) -> Dict[str, Pose]:
        q_dict = self._q_to_dict(q)
        T_map = self.robot.link_fk(cfg=q_dict, use_names=True)
        frames: Dict[str, Pose] = {}
        for link, T in T_map.items():
            name = link
            if name is None:
                continue
            frames[name] = T_to_pose(T)
        return frames

    # -------------------------------------------------------------------------
    # Jacobian
    # -------------------------------------------------------------------------
    def jacobian(
        self,
        q: np.ndarray,
        link_name: Optional[str] = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "URDFPy backend does not support Jacobian computation. "
            "Please use a different backend (e.g., Pinocchio) for Jacobian capabilities."
        )
        
    # -------------------------------------------------------------------------
    # IK (Not Supported)
    # -------------------------------------------------------------------------
    def ik(
        self,
        target_pose: Pose,
        initial_joint_positions: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Inverse kinematics is not supported by urdfpy backend.
        
        Raises:
            NotImplementedError: Always, as urdfpy does not provide IK capabilities.
        """
        raise NotImplementedError(
            "URDFPy backend does not support inverse kinematics (IK). "
            "Please use a different backend (e.g., Pinocchio or RelaxIK) for IK capabilities."
        )
