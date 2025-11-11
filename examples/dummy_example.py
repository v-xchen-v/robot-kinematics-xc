from robot_kinematics.backends.dummy_backend import DummyKinematicsBackend
from typing import Any, Dict, Optional
import numpy as np
from robot_kinematics.core.types import Pose, IKResult, IKOptions

if __name__ == "__main__":
    # Simple test of the DummyKinematicsBackend
    backend = DummyKinematicsBackend()

    # Test FK
    q_test = np.array([0.0, 0.0, 0.0])
    pose = backend.fk(q_test, backend.base_link, "ee_link")
    print(f"FK at q={q_test}: position={pose.xyz}, orientation={pose.quat_wxyz}")

    # Test IK
    target_pose = Pose(xyz=np.array([2.0, 0.0, 0.0]), quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))
    ik_options = IKOptions(max_iters=100, tol=1e-4, extra={"step_size": 0.5})
    ik_result = backend.ik(target_pose, seed_q=None, options=ik_options, base=backend.base_link, link="ee_link")
    print(f"IK result: success={ik_result.success}, q={ik_result.q}, pos_err={ik_result.pos_err}")