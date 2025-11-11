from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..core.types import Pose

def T_to_pose(T: np.ndarray) -> Pose:
    """Convert a 4x4 transformation matrix to a Pose object."""
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # returns in (x, y, z, w) format
    # Convert to (w, x, y, z) format
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    return Pose(xyz=position, quat_wxyz=quaternion)

def pose_to_T(pose: Pose) -> np.ndarray:
    """Convert a Pose object to a 4x4 transformation matrix."""
    w, x, y, z = pose.quat_wxyz
    rotation = R.from_quat([x, y, z, w], scalar_first=False)
    T_R = rotation.as_matrix()
    T = np.eye(4)
    T[:3, :3] = T_R
    T[:3, 3] = pose.xyz
    return T
