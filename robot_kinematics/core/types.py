# robot_kinematics_xc/core/types.py
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class Pose:
    """Represents a 3D pose with position and orientation.
    
    Attributes:
        position: 3D position vector [x, y, z]
        orientation: Quaternion [w, x, y, z] in scalar-first format
    """
    xyz: np.ndarray     # shape: (3,)
    quat_wxyz: np.ndarray  # shape: (4,) quaternion (w, x, y, z) scalar-first
    
    def as_matrix(self) -> np.ndarray:
        """Convert pose to a 4x4 transformation matrix."""
        w, x, y, z = self.quat_wxyz
        T_R = R.from_quat([x, y, z, w], scalar_first=False).as_matrix()
        T = np.eye(4)
        T[:3, :3] = T_R
        T[:3, 3] = self.xyz
        return T
    
    def as_flat_array(self) -> np.ndarray:
        """Convert pose to a flat array [x, y, z, qw, qx, qy, qz]."""
        return np.concatenate((self.xyz, self.quat_wxyz))


@dataclass
class IKOptions:
    max_iters: int = 100
    tol: float = 1e-4
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    regularization_weight: float = 1e-4
    collision_fn: Optional[Callable[[np.ndarray], bool]] = None

    # NEW: backend-specific options
    extra: Dict[str, Any] = field(default_factory=dict)
    """
    Backend-specific options, interpreted by each backend however it likes.
    E.g. {'pinocchio_damping': 1e-3, 'relaxed_ik_use_collisions': True}
    """
    

@dataclass
class IKResult:
    """
    Inverse Kinematics result.
    - success: True if solver converged below tolerance
    - q: final joint configuration (None if failed)
    - pos_err: final position error (m)
    - ori_err: final orientation error (radians)
    - err: combined weighted total error (sqrt(pos_err² + ori_err²))
    - info: optional backend-specific debug info
    """
    success: bool
    q: Optional[np.ndarray] = None
    pos_err: float = np.inf
    ori_err: float = np.inf
    info: Optional[Any] = field(default_factory=dict)