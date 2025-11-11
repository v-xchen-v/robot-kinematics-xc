# Inverse Kinematics (IK) Implementation

## Overview

The IK implementation in the Pinocchio backend uses **CasADi** optimization library with **IPOPT** solver to find joint configurations that achieve desired end-effector poses. This approach is based on constrained nonlinear optimization and is more robust than iterative Jacobian-based methods.

## Key Features

- **Optimization-based**: Uses CasADi with IPOPT for constrained nonlinear optimization
- **Joint limits**: Automatically respects joint position limits from the URDF
- **Robust convergence**: Includes acceptable solution mechanism for better convergence
- **Efficient**: Lazy initialization of solver (only created when first needed)
- **Flexible**: Supports custom solver options and parameters

## Installation Requirements

To use IK functionality, you need to install CasADi:

```bash
pip install casadi
```

For the Pinocchio backend with CasADi support:

```bash
conda install pinocchio -c conda-forge
pip install casadi
```

## Mathematical Formulation

### Optimization Problem

The IK problem is formulated as:

```
minimize    ||log6(T_target^-1 * T_ee(q))||^2
subject to  q_min ≤ q ≤ q_max
```

Where:
- `q` is the joint configuration vector
- `T_target` is the desired end-effector pose (4×4 transformation matrix)
- `T_ee(q)` is the actual end-effector pose at configuration q
- `log6` is the SE(3) logarithm map (converts pose error to 6D vector)
- `q_min`, `q_max` are joint position limits from URDF

### Error Metric

The error function uses Pinocchio's `log6` function which computes the logarithm of the SE(3) transformation:

```python
error = log6(T_target^-1 * T_ee(q))
```

This gives a 6D vector representing:
- Linear velocity (3D): position error
- Angular velocity (3D): orientation error

## Usage

### Basic Usage

```python
from robot_kinematics.backends.pinocchio_backend import PinocchioKinematicsBackend
from robot_kinematics.frames.transforms import Pose
import numpy as np

# Initialize backend
backend = PinocchioKinematicsBackend(
    urdf_path="path/to/robot.urdf",
    base_link="base_link",
    ee_link="end_effector_link",
    package_dirs=["path/to/meshes"],
)

# Define target pose
target_pose = Pose(
    position=np.array([0.5, 0.2, 0.3]),
    quaternion=np.array([0.0, 0.0, 0.0, 1.0])  # [qx, qy, qz, qw]
)

# Solve IK
initial_guess = np.zeros(backend.n_dof)
joint_solution = backend.ik(
    target_pose=target_pose,
    initial_joint_positions=initial_guess,
    max_iterations=200,
    tolerance=1e-4
)

print(f"Solution: {joint_solution}")
```

### Advanced Usage with Success Checking

```python
# Get detailed results including convergence status
solution_q, success, achieved_pose = backend.ik(
    target_pose=target_pose,
    initial_joint_positions=initial_guess,
    return_success=True,  # Enable returning success info
    max_iterations=200,
    tolerance=1e-4
)

print(f"Converged: {success}")
print(f"Solution: {solution_q}")
print(f"Achieved position: {achieved_pose.position}")
print(f"Target position: {target_pose.position}")

# Calculate position error
position_error = np.linalg.norm(target_pose.position - achieved_pose.position)
print(f"Position error: {position_error:.6f} m")
```

### Custom Solver Options

```python
# Define custom IPOPT options
custom_options = {
    'ipopt': {
        'print_level': 5,  # Verbose output
        'max_iter': 500,   # More iterations
        'tol': 1e-6,       # Tighter tolerance
        'linear_solver': 'ma57',  # Different linear solver
    },
    'print_time': True
}

# Solve with custom options
joint_solution = backend.ik(
    target_pose=target_pose,
    solver_options=custom_options
)
```

## Solver Options

### Default IPOPT Options

The default solver configuration includes:

```python
{
    'ipopt': {
        'print_level': 0,                    # Silent mode
        'max_iter': 200,                     # Maximum iterations
        'tol': 1e-4,                         # Main tolerance
        'constr_viol_tol': 1e-4,            # Constraint violation tolerance
        'dual_inf_tol': 1e-4,               # Dual infeasibility tolerance
        'compl_inf_tol': 1e-4,              # Complementarity tolerance
        
        # Acceptable solution mechanism
        'acceptable_tol': 1e-3,             # Relaxed tolerance
        'acceptable_iter': 15,              # Accept after 15 iterations
        'acceptable_constr_viol_tol': 1e-3,
        'acceptable_dual_inf_tol': 1e-3,
        'acceptable_compl_inf_tol': 1e-3,
        
        # Numerical stability
        'mu_strategy': 'adaptive',
        'linear_solver': 'mumps',
        'hessian_approximation': 'limited-memory',
        'check_derivatives_for_naninf': 'yes',
        
        # Numerical tolerance
        'bound_relax_factor': 1e-8,
        'honor_original_bounds': 'yes',
        'nlp_scaling_method': 'gradient-based',
    },
    'print_time': False
}
```

### Key Parameters Explained

- **max_iter**: Maximum number of optimization iterations. Increase for complex problems.
- **tol**: Main convergence tolerance. Smaller values give more accurate solutions but may fail to converge.
- **acceptable_tol**: Relaxed tolerance for acceptable solutions. Helps convergence for difficult problems.
- **acceptable_iter**: Number of consecutive iterations meeting acceptable tolerance before termination.
- **linear_solver**: Linear algebra solver. Options: 'mumps' (default), 'ma27', 'ma57', 'ma86', 'ma97'.
- **hessian_approximation**: Use 'limited-memory' (L-BFGS) for faster convergence with large systems.

## Troubleshooting

### Problem: IK doesn't converge

**Solutions:**
1. Increase `max_iter` (e.g., 500 or 1000)
2. Relax `tolerance` (e.g., 1e-3 instead of 1e-4)
3. Provide better initial guess (use FK result from nearby configuration)
4. Check if target pose is reachable (workspace limits)

### Problem: Solution is inaccurate

**Solutions:**
1. Decrease `tolerance` (e.g., 1e-5 or 1e-6)
2. Use tighter acceptable tolerances
3. Increase `max_iter` to allow more time for convergence
4. Check joint limits in URDF are correct

### Problem: Slow performance

**Solutions:**
1. Use 'limited-memory' Hessian approximation
2. Reduce `max_iter` if accuracy is not critical
3. Try different linear solvers (ma57 is often faster than mumps)
4. Warm-start with previous solution as initial guess

### Problem: CasADi import error

**Solution:**
```bash
pip install casadi
```

If that fails, try:
```bash
conda install -c conda-forge casadi
```

## Implementation Details

### Lazy Initialization

The IK solver is initialized lazily (on first use) to avoid overhead when only FK is needed:

```python
if not self._ik_solver_initialized:
    self._initialize_ik_solver(solver_options)
```

### CasADi Model Creation

The implementation creates a CasADi model from Pinocchio:

```python
cmodel = cpin.Model(self.model)
cdata = cmodel.createData()
cq = casadi.SX.sym("q", self.model.nq, 1)
cTf = casadi.SX.sym("tf", 4, 4)
```

### Error Function

The error function uses SE(3) logarithm:

```python
error = casadi.Function(
    "error",
    [cq, cTf],
    [cpin.log6(cdata.oMf[ee_frame_id].inverse() * cpin.SE3(cTf)).vector]
)
```

## Comparison with Other Methods

### CasADi Optimization vs Jacobian-based

| Aspect | CasADi Optimization | Jacobian-based |
|--------|-------------------|----------------|
| Convergence | More robust | Can fail for singular configurations |
| Speed | Slower per solve | Faster per iteration |
| Joint limits | Automatic handling | Requires clamping |
| Accuracy | High | Depends on step size |
| Implementation | Complex setup, simple usage | Simple setup, complex tuning |

### When to Use CasADi

- Need guaranteed joint limit satisfaction
- Require high accuracy
- Have complex constraints
- Can afford computation time

### When to Use Jacobian-based

- Need real-time performance
- Configuration is far from singularities
- Simple joint limit handling is sufficient
- Have good initial guess

## References

1. **Pinocchio Documentation**: https://stack-of-tasks.github.io/pinocchio/
2. **CasADi Documentation**: https://web.casadi.org/
3. **IPOPT Documentation**: https://coin-or.github.io/Ipopt/
4. **SE(3) Logarithm**: Murray, R. M., Li, Z., & Sastry, S. S. (1994). A mathematical introduction to robotic manipulation.

## Examples

See the following example files:
- `examples/pinocchio_ik_example.py` - Basic IK usage
- `examples/a2d_arm_ik_example.py` - Advanced A2D robot IK solver
- `examples/pinocchio_backend_example.py` - General backend usage

## License

This implementation is part of the robot-kinematics package and follows the same MIT license.
