from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

def T_to_pose(T: np.ndarray) -> Pose:
    """Convert a 4x4 transformation matrix to a Pose object."""
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # returns in (x, y, z, w) format
    # Convert to (w, x, y, z) format
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    return Pose(position=position, orientation=quaternion)

def pose_to_T(pose: Pose) -> np.ndarray:
    """Convert a Pose object to a 4x4 transformation matrix."""
    w, x, y, z = pose.orientation
    rotation = R.from_quat([x, y, z, w], scalar_first=False)
    T_R = rotation.as_matrix()
    T = np.eye(4)
    T[:3, :3] = T_R
    T[:3, 3] = pose.position
    return T

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


@dataclass
class PoseDelta:
    """Represents a displacement/difference between two 3D poses.
    
    Attributes:
        position_delta: 3D position displacement [dx, dy, dz]
        orientation_delta: Quaternion displacement [dqw, dqx, dqy, dqz]
    """
    position_delta: np.ndarray      # shape: (3,) - position displacement
    orientation_delta: np.ndarray   # shape: (4,) - quaternion displacement
    
    def as_flat_array(self) -> np.ndarray:
        """Convert to flat array [dx, dy, dz, dqw, dqx, dqy, dqz]."""
        return np.concatenate((self.position_delta, self.orientation_delta))
    
    @classmethod
    def from_poses(cls, pose1: Pose, pose2: Pose) -> "PoseDelta":
        """Compute pose delta from two poses (pose2 - pose1).
        
        Args:
            pose1: Starting pose
            pose2: Target pose
            
        Returns:
            PoseDelta representing the displacement
        """
        position_delta = pose2.position - pose1.position
        orientation_delta = pose2.orientation - pose1.orientation
        
        return cls(
            position_delta=position_delta,
            orientation_delta=orientation_delta
        )
    
    def apply_to_pose(self, pose: Pose) -> Pose:
        """Apply this delta to a pose to get a new pose.
        
        Args:
            pose: Base pose to apply delta to
            
        Returns:
            New pose with delta applied
        """
        new_position = pose.position + self.position_delta
        new_orientation = pose.orientation + self.orientation_delta
        
        # Normalize the quaternion to ensure it's valid
        new_orientation = new_orientation / np.linalg.norm(new_orientation)
        
        return Pose(position=new_position, orientation=new_orientation)
