# robot_kinematics_xc/core/types.py
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict
import numpy as np

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