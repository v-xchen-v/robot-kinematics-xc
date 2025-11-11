from robot_kinematics.core.robot_kinematics import RobotKinematics
import numpy as np
from robot_kinematics.core.types import Pose, IKOptions

def main():
    urdf_path = "../robots/g1/A2D_120s/urdf/A2D.urdf"
    
# Change backend_name here: "dummy", "urdfpy", "pinocchio", etc
    robot = RobotKinematics(
        urdf_path=urdf_path,
        base_link="base_link",
        ee_link="gripper_r_center_link",
        backend="dummy",
    )
    
     # ---------- FK with array ----------
    q_home = np.zeros(robot.n_dofs)
    pose_home = robot.fk(q_home)
    print("\nFK (array) - home pose:")
    print("  position   :", pose_home.xyz)
    print("  orientation:", pose_home.quat_wxyz)
    
    # ---------- FK with dict subset ----------
    # Only specify some joints, others use default_q
    q_cfg = {robot.joint_names[0]: 0.2, robot.joint_names[2]: -0.4}
    pose_cfg = robot.fk(q_cfg)
    print("\nFK (dict subset):")
    print("  position   :", pose_cfg.xyz)
    
     # ---------- IK ----------
    target_pose = Pose(
        xyz=pose_home.xyz + np.array([0.05, 0.0, 0.05]),
        quat_wxyz=pose_home.quat_wxyz,
    )

    options = IKOptions(
        max_iters=150,
        tol=1e-4,
        extra={"damping": 1e-3},   # backend-specific if supported
    )

    ik_result = robot.ik(target=target_pose, seed_q=q_home, options=options)

    print("\nIK Result:")
    print("  success :", ik_result.success)
    print("  pos_err :", ik_result.pos_err)
    print("  ori_err :", ik_result.ori_err)

    if ik_result.q is not None:
        print("  q (array):", ik_result.q)
        print("  q (dict) :")
        for name, val in robot.q_array_to_dict(ik_result.q).items():
            print(f"    {name:<20} {val:.3f}")
    
    
main()