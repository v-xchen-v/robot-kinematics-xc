from typing import Any
from .base_kinematics_backend import BaseKinematicsBackend
def get_kinematics_backend(
    backend_name: str,
    **kwargs: Any
) -> BaseKinematicsBackend:
    """
    Factory function to get the appropriate kinematics backend instance.
    
    Args:
        backend_name: Name of the backend ("pinocchio", "urdfpy", etc.)
        **kwargs: Additional arguments specific to the backend.
    """
    backend_name = backend_name.lower()
    if backend_name == "pinocchio":
        from .pinocchio_backend import PinocchioKinematicsBackend
        return PinocchioKinematicsBackend(**kwargs)
    elif backend_name == "urdfpy":
        from .urdfpy_backend import URDFPyKinematicsBackend
        return URDFPyKinematicsBackend(**kwargs)
    elif backend_name == "dummy":
        from .dummy_backend import DummyKinematicsBackend
        return DummyKinematicsBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")