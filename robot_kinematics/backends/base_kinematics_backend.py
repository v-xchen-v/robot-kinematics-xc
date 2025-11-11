from __future__ import annotations

from typing import Dict, Optional, Any, List, Mapping, Union
import numpy as np
from abc import ABC, abstractmethod

from ..frames import Pose
from ..core.types import IKOptions
from ..urdf.inspector import SubchainURDFInspector
from ..core.types import IKResult

JointCfg = Mapping[str, float]

def build_backend(
    backend_name: str, 
    urdf_path: str,
    base_link: str,
    ee_link: str,
    **kwargs) -> BaseKinematicsBackend:
    """
    Factory function to build a kinematics backend instance.

    Args:
        backend_name: str
            Name of the backend to use (e.g., 'urdfpy', 'pinocchio', 'relaxik').
        **kwargs: Any
            Additional keyword arguments to pass to the backend constructor.

    Returns:
        BaseKinematicsBackend
            An instance of the requested kinematics backend.
    """
    if backend_name == 'urdfpy':
        from .urdfpy_backend import URDFPyKinematicsBackend
        return URDFPyKinematicsBackend(
            urdf_path=urdf_path,
            base_link=base_link,
            ee_link=ee_link,
            **kwargs
        )
    elif backend_name == 'pinocchio':
        from .pinocchio_backend import PinocchioKinematicsBackend
        return PinocchioKinematicsBackend(
            urdf_path=urdf_path,
            base_link=base_link,
            ee_link=ee_link,
            **kwargs
        )
    # elif backend_name == 'relaxik':
    #     from .relaxik_backend import RelaxIKKinematicsBackend
    #     return RelaxIKKinematicsBackend(**kwargs)
    else:
        raise ValueError(f"Unknown kinematics backend: {backend_name}")
    
class BaseKinematicsBackend(ABC):
    """
    Abstract base class for kinematics backends.
    
    This class provides a common interface for FK/IK/Jacobian computations
    and shared utility methods using the URDF inspector.
    
    Subclasses (URDFPyKinematicsBackend, PinocchioKinematicsBackend, etc.) 
    should:
    1. Initialize self._urdf_inspector in their __init__
    2. Set self.urdf_path, self.base_link, self.ee_link
    3. Implement abstract methods (fk, ik, etc.)
    
    The shared methods (list_links, list_joints) will then work automatically
    through inheritance.
    """
    def __init__(self):
        """
        Base initializer. Subclasses should call this after setting up
        their own attributes, or initialize _urdf_inspector directly.
        """
        # Initialize the inspector as None - subclasses should set this
        self._urdf_inspector = None
    
    # ------------------------------------------------------------------
    # Abstract methods that subclasses must implement
    # ------------------------------------------------------------------
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
        
    @abstractmethod
    def ik(
        self,
        target_pose: Pose,
        initial_joint_positions: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> IKResult:
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
        
    # ------------------------------------------------------------------
    # Shared utility methods using URDF inspector
    # These methods work automatically in subclasses if they initialize
    # self._urdf_inspector properly
    # ------------------------------------------------------------------
    def _get_urdf_inspector(self) -> Optional[SubchainURDFInspector]:
        """
        Get the cached URDF inspector instance.
        
        Subclasses should initialize self._urdf_inspector in their __init__.
        
        Returns:
            SubchainURDFInspector or None: The inspector if initialized, None otherwise.
        """
        return self._urdf_inspector
    
    def list_links(self) -> List[str]:
        """
        Get the list of frame/link names in the robot model between base_link and end_effector_link.
        
        This method uses the SubchainURDFInspector to traverse the URDF and find all links
        in the kinematic chain. Subclasses can override this method to provide backend-specific
        implementations if needed.

        Returns:
            List[str]: List of link/frame names in the kinematic chain.
                      Returns empty list if inspector is not initialized.
        """
        inspector = self._get_urdf_inspector()
        if inspector is not None:
            return inspector.list_links()
        return []
    
    def list_joints(self, movable_only: bool = True) -> List[str]:
        """
        Get the list of joint names in the robot model between base_link and end_effector_link.
        
        This method uses the SubchainURDFInspector to traverse the URDF and find all joints
        in the kinematic chain. Subclasses can override this method to provide backend-specific
        implementations if needed.

        Args:
            movable_only: bool, optional
                If True, return only movable joints (revolute, prismatic, continuous).
                If False, return all joints including fixed joints.
                Default is True.
                
        Returns:
            List[str]: List of joint names in the kinematic chain.
                      Returns empty list if inspector is not initialized.
        """
        inspector = self._get_urdf_inspector()
        if inspector is not None:
            return inspector.get_joint_names(movable_only=movable_only)
        return []