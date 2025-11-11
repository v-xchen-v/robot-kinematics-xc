# robot_kinematics_xc/backends/dummy_backend.py

import numpy as np
from robot_kinematics.core.types import Pose, IKResult, IKOptions
from .base_kinematics_backend import BaseKinematicsBackend


class DummyKinematicsBackend(BaseKinematicsBackend):
    """
    Dummy kinematics backend.

    - 3-DOF planar arm in XY plane
    - link lengths: [1.0, 1.0, 1.0]
    - base at origin, end-effector at end of third link
    """

    def __init__(self, urdf_path: str = "", base_link: str = "base_link", **kwargs):
        super().__init__()

        self.base_link = base_link
        self.link_lengths = np.array([1.0, 1.0, 1.0], dtype=float)

        # Public contract
        self.joint_names = ["joint1", "joint2", "joint3"]
        self.n_dofs = len(self.joint_names)
        self.default_q = np.zeros(self.n_dofs, dtype=float)

    # ---------------------- FK ---------------------- #
    def fk(self, q: np.ndarray, base: str, link: str) -> Pose:
        """
        Simple planar FK:
            x = Σ L_i cos(Σ_j q_j)
            y = Σ L_i sin(Σ_j q_j)
            z = 0
        Orientation is fixed identity quaternion.
        """
        q = np.asarray(q, dtype=float)
        assert q.shape == (self.n_dofs,)

        # cumulative joint angles
        cum_angles = np.cumsum(q)
        x = np.sum(self.link_lengths * np.cos(cum_angles))
        y = np.sum(self.link_lengths * np.sin(cum_angles))

        position = np.array([x, y, 0.0], dtype=float)
        orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # w, x, y, z

        return Pose(xyz=position, quat_wxyz=orientation)

    # ---------------------- IK ---------------------- #
    def ik(
        self,
        target: Pose,
        seed_q: np.ndarray,
        options: IKOptions,
        base: str,
        link: str,
    ) -> IKResult:
        """
        Position-only IK using very simple gradient-like updates.

        - Ignores orientation (ori_err always 0).
        - Uses numeric Jacobian in XY.
        """
        q = seed_q.copy() if seed_q is not None else self.default_q.copy()
        step_size = options.extra.get("step_size", 0.5)
        eps = 1e-6

        pos_err_val = np.inf
        ori_err_val = 0.0
        success = False
        iters = 0

        for it in range(options.max_iters):
            iters = it + 1

            pose = self.fk(q, base, link)
            diff_xy = target.xyz[:2] - pose.xyz[:2]
            pos_err_val = float(np.linalg.norm(diff_xy))

            if pos_err_val < options.tol:
                success = True
                break

            # numeric Jacobian in XY: J_xy is (2, dof)
            J_xy = np.zeros((2, self.n_dofs), dtype=float)
            for j in range(self.n_dofs):
                dq = np.zeros_like(q)
                dq[j] = eps
                pose_pert = self.fk(q + dq, base, link)
                diff_pert = pose_pert.xyz[:2] - pose.xyz[:2]
                J_xy[:, j] = diff_pert / eps

            # damped least-squares
            H = J_xy.T @ J_xy + options.regularization_weight * np.eye(self.n_dofs)
            g = J_xy.T @ diff_xy
            dq = step_size * np.linalg.solve(H, g)
            q = q + dq

            # optional collision check
            if options.collision_fn is not None and options.collision_fn(q):
                return IKResult(
                    success=False,
                    q=q,
                    pos_err=pos_err_val,
                    ori_err=0.0,
                    info={"reason": "collision", "iters": iters},
                )

        return IKResult(
            success=success,
            q=q,
            pos_err=pos_err_val,
            ori_err=ori_err_val,
            info={"iters": iters},
        )
        

    def fk_all_frames(self, joint_positions):
        return super().fk_all_frames(joint_positions)