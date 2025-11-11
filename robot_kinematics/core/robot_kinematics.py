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
        extra: Optional[dict] = None,  # backend-specific init args
    ):
        from robot_kinematics.backends import get_kinematics_backend
        self._backend = get_kinematics_backend(
            backend,
            urdf_path=urdf_path,
            base_link=base_link,
            ee_link=ee_link,
            **(extra or {}),
        )
        
        self.base_link = base_link
        self.ee_link = ee_link

        self.joint_names = list(self._backend.joint_names)
        self.n_dofs = self._backend.n_dofs
        self._name_to_index = {n: i for i, n in enumerate(self.joint_names)}
        
    # ---------------------- helpers ---------------------- #
    def q_array_to_dict(self, q: np.ndarray) -> dict:
        """Convert joint vector → {joint_name: value}."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_dofs,):
            raise ValueError(f"Expected q shape ({self.dof},), got {q.shape}")
        return {name: float(q[i]) for i, name in enumerate(self.joint_names)}

    def q_dict_to_array(self, q_dict: dict) -> np.ndarray:
        """Convert {joint_name: value} (subset or full) → joint vector."""
        q = np.zeros(self.n_dofs, dtype=float)
        for name, value in q_dict.items():
            if name not in self._name_to_index:
                raise KeyError(f"Unknown joint name: {name}")
            q[self._name_to_index[name]] = float(value)
        return q

    def _q_from_array_or_cfg(self, q_or_cfg: Union[np.ndarray, JointCfg]) -> np.ndarray:
        """Internal: accept array or cfg dict, always return full array."""
        if isinstance(q_or_cfg, np.ndarray):
            q = np.asarray(q_or_cfg, dtype=float)
            if q.shape != (self.n_dofs,):
                raise ValueError(f"Expected q shape ({self.n_dofs},), got {q.shape}")
            return q
        return self.q_dict_to_array(q_or_cfg)
    
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
            seed_q = self.q_dict_to_array(seed_q)

        result = self._backend.ik(
            target_pose=target_pose,
            seed_q=seed_q,
            options=options,
            base=self.base_link,
            link=self.ee_link,
        )
        # result is an IKResult already
        return result
