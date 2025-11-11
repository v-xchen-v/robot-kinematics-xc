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
        self._urdf_inspector = None  # Lazy-initialized cache
        
        
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
    def _get_urdf_inspector(self):
        """
        Lazy-initialize and cache the URDF inspector.
        
        Returns:
            SubchainURDFInspector or None
        """
        if self._urdf_inspector is None:
            if hasattr(self, 'urdf_path') and hasattr(self, 'base_link') and hasattr(self, 'ee_link'):
                from ..urdf.inspector import SubchainURDFInspector
                self._urdf_inspector = SubchainURDFInspector(self.urdf_path, self.base_link, self.ee_link)
        return self._urdf_inspector
    
    def list_links(self) -> List[str]:
        """
        Get the list of frame/link names in the robot model between base_link and end_effector_link.
        
        Default implementation uses SubchainURDFInspector if urdf_path, base_link, and ee_link
        attributes are available. Otherwise returns an empty list. Subclasses can override this
        method to provide backend-specific implementations.

        Returns:
            List[str]
                List of frame/link names.
        """
        inspector = self._get_urdf_inspector()
        if inspector is not None:
            return inspector.list_links()
        return []
    
    def list_joints(self, movable_only: bool = True) -> List[str]:
        """
        Get the list of joint names in the robot model between base_link and end_effector_link.
        
        Default implementation uses SubchainURDFInspector if urdf_path, base_link, and ee_link
        attributes are available. Otherwise returns an empty list. Subclasses can override this
        method to provide backend-specific implementations.

        Args:
            movable_only: bool, optional
                If True, return only movable joints. Default is True.   
        Returns:
            List[str]
                List of joint names.
        """
        inspector = self._get_urdf_inspector()
        if inspector is not None:
            return inspector.get_joint_names(movable_only=movable_only)
        return []
    
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