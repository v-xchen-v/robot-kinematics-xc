# The Client/User SDK class
# cores/robot_kinematics.py

from typing import Optional, Mapping, Union
import numpy as np
from .types import Pose
from .types import IKResult, IKOptions

JointCfg = Mapping[str, float]

class RobotKinematics:
    """
    Unified FK/IK Interface.
    
    - Initialize directly with URDF and backend name.
    - FK accepts np.array or dict {joint_name: value}
    - IK returns IKResult with a q as np.ndarray in backend joint order.
    """
    
    def __init__(
        self,
        urdf_path: str,
        base_link: str,
        ee_link: str,
        backend: str = "pinocchio",
        active_joints: Optional[list] = None, # If provided, only these joints are considered active
        inactive_joints_seed: Optional[Mapping[str, float]] = None, # Should be provided if active_joints is subset of chain joints, inactive joints are fixed to these values during IK
        extra: Optional[dict] = None,  # backend-specific init args
    ):
        from robot_kinematics.backends import get_kinematics_backend
        self._backend = get_kinematics_backend(
            backend,
            urdf_path=urdf_path,
            base_link=base_link,
            ee_link=ee_link,
            active_joints=active_joints,
            **(extra or {}),
        )
        
        self.base_link = base_link
        self.ee_link = ee_link

        self.joint_names = list(self._backend.joint_names)
        self.n_dofs = self._backend.n_dofs
        self._name_to_index = {n: i for i, n in enumerate(self.joint_names)}
        self.active_joint_names = self._backend.active_joint_names
        self.n_active_dofs = len(self.active_joint_names) if self.active_joint_names else self.n_dofs
        self.active_joint_indices = self._backend.active_joint_indices
        
    # ---------------------- helpers ---------------------- #
    def q_array_to_dict(self, q: np.ndarray) -> dict:
        """Convert joint vector → {joint_name: value}."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_active_dofs,):
            raise ValueError(f"Expected q shape ({self.dof},), got {q.shape}")
        return {name: float(q[i]) for i, name in enumerate(self.active_joint_names)}

    def q_dict_to_array(self, q_dict: dict, active_only=False) -> np.ndarray:
        """Convert {joint_name: value} (subset or full) → joint vector."""
        q = np.zeros(self.n_dofs, dtype=float)
        for name, value in q_dict.items():
            if name not in self._name_to_index:
                raise KeyError(f"Unknown joint name: {name}")
            q[self._name_to_index[name]] = float(value)
        if active_only:
            return q[self.active_joint_indices]
        return q

    def _q_from_array_or_cfg(self, q_or_cfg: Union[np.ndarray, JointCfg]) -> np.ndarray:
        """Internal: accept array or cfg dict, always return full array."""
        if isinstance(q_or_cfg, np.ndarray):
            q = np.asarray(q_or_cfg, dtype=float)
            if q.shape != (self.n_active_dofs,):
                raise ValueError(f"Expected q shape ({self.n_active_dofs},), got {q.shape}")
            return q
        return self.q_dict_to_array(q_or_cfg)
    
    def filter_q(self, q_full: np.ndarray) -> np.ndarray:
        """Filter a full joint vector to only include active joints."""
        return q_full[self.active_joint_indices]

    def expand_q_full(
        self, 
        q_active: Mapping[str, float], 
        q_full_ref: np.ndarray, 
        use_joint_names=False
    ) -> Union[np.ndarray, Mapping[str, float]]:
        """Expand active joints into a full joint vector, keeping others fixed."""
        q_full = q_full_ref.copy()
        for i, idx in enumerate(q_active.keys()):
            joint_index = self._name_to_index[idx]
            q_full[joint_index] = q_active[idx]
            
        if use_joint_names:
            q_full = {name: q_full[i] for i, name in enumerate(self.joint_names)}
        return q_full
    
        # -------------------------- FK -------------------------- #
    def fk(self, joint_positions: Union[np.ndarray, JointCfg], target_link: Optional[str] = None) -> Pose:
        """
        Compute FK in base_link frame.

        Args:
            q:  np.ndarray (dof,) OR dict {joint_name: value}
            link: target link; default = self.ee_link
        """
        q_vec = self._q_from_array_or_cfg(joint_positions)
        target_link = target_link or self.ee_link
        return self._backend.fk(joint_positions=q_vec, target_link=target_link)

    # -------------------------- IK -------------------------- #
    def ik(
        self,
        target_pose: Pose,
        seed_q: Optional[Union[np.ndarray, JointCfg]] = None,
        options: Optional[IKOptions] = None,
    ) -> IKResult:
        """
        Solve IK to reach target Pose (in base_link frame).

        seed_q: initial joint position guess for IK solver.
        seed_q can be:
            - np.ndarray in backend joint order
            - dict {joint_name: value}
            - None (backend uses its own default)
        """
        options = options or IKOptions()

        if isinstance(seed_q, dict):
            seed_q = self.q_dict_to_array(seed_q, active_only=True)

        result = self._backend.ik(
            target_pose=target_pose,
            seed_q=seed_q,
            options=options,
            base=self.base_link,
            link=self.ee_link,
        )
        # result is an IKResult already
        return result
