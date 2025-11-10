from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Pose:
    """Represents a 3D pose with position and orientation.
    
    Attributes:
        position: 3D position vector [x, y, z]
        orientation: Quaternion [w, x, y, z] in scalar-first format
    """
    position: np.ndarray     # shape: (3,)
    orientation: np.ndarray  # shape: (4,) quaternion (w, x, y, z) scalar-first
    
    def as_matrix(self) -> np.ndarray:
        """Convert pose to a 4x4 transformation matrix."""
        w, x, y, z = self.orientation
        T_R = R.from_quat([x, y, z, w], scalar_first=False).as_matrix()
        T = np.eye(4)
        T[:3, :3] = T_R
        T[:3, 3] = self.position
        return T
    
    def as_flat_array(self) -> np.ndarray:
        """Convert pose to a flat array [x, y, z, qw, qx, qy, qz]."""
        return np.concatenate((self.position, self.orientation))
