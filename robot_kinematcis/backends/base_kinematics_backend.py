from __future__ import annotations

from typing import Dict, Optional, Any, List
import numpy as np
from abc import ABC, abstractmethod

from ..frames import Pose
    
    
class BaseKinematicsBackend(ABC):
    """
    Backend interface for FK/IK/Jacobian computations.
    Concrete implementations should inherit from this class and implement the abstract methods with urdfpy, pinocchio, RelaxIK, etc.
    """
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata if metadata is not None else {}
        
    # --- Constuction --- #
    @classmethod
    @abstractmethod
    def from_urdfpy(
        cls,
        urdf_path: str,
        base_link: str,
        end_effector_link: str,
        joint_names: Optional[List[str]] = None,
        **kwargs: Any
    ) -> "BaseKinematicsBackend":
        """
        Create a kinematics backend from a URDF file.
        
        Args:
            urdf_path: str
                Path to the URDF file.
            base_link: str
                Name of the base link.
            end_effector_link: str
                Name of the end-effector link.
            joint_names: Optional[List[str]], optional
                Subset/ordering of joints we care about. If None,
                use all movable joints in URDF order.
            **kwargs: Any
                Additional backend-specific arguments.
        """
        ...
        
        
    # --- Forward Kinematics --- #
    @abstractmethod
    def fk(
        self,
        joint_positions: np.ndarray,
        target_link: Optional[str] = None,
    ) -> Pose:
        """
        Compute the forward kinematics to get the target link pose.

        Args:
            joint_positions: np.ndarray, shape (n_joints,)
                Joint positions.
            target_link: Optional[str], optional
                Name of the target link to compute pose for.
                If None, use the default end-effector link configured at construction.

        Returns:
            Pose
                The target link pose.
        """
        ...
        
    # --- Inverse Kinematics --- #
    @abstractmethod
    def ik(
        self,
        target_pose: Pose,
        initial_joint_positions: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute the inverse kinematics to get joint positions for the desired target pose.

        Args:
            target_pose: Pose
                Desired target pose for the end-effector.
            initial_joint_positions: Optional[np.ndarray], optional
                Initial guess for joint positions. If None, use zeros.
            **kwargs: Any
                Additional backend-specific arguments.

        Returns:
            np.ndarray
                Joint positions that achieve the desired target pose.
        """
        ...
        
        
    # --- Jacobian --- #
    @abstractmethod
    def jacobian(
        self,
        joint_positions: np.ndarray,
        target_link: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix at the given joint positions.

        Args:
            joint_positions: np.ndarray, shape (n_joints,)
                Joint positions.
            target_link: Optional[str], optional
                Name of the target link to compute Jacobian for.
                If None, use the default end-effector link configured at construction.

        Returns:
            np.ndarray
                The Jacobian matrix of shape (6, n_joints).
        """
        ...
        
    # --- Additional Methods --- #
    @abstractmethod
    def get_frame_names(self) -> List[str]:
        """
        Get the list of frame/link names in the robot model.

        Returns:
            List[str]
                List of frame/link names.
        """
        ...
        
    @abstractmethod
    def fk_all_frames(
        self,
        joint_positions: np.ndarray,
    ) -> Dict[str, Pose]:
        """
        Compute forward kinematics for all frames/links in the robot model.

        Args:
            joint_positions: np.ndarray, shape (n_joints,)
                Joint positions.

        Returns:
            Dict[str, Pose]
                Dictionary mapping frame/link names to their poses.
        """
        ...