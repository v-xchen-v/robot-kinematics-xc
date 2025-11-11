# Examples

This directory contains example scripts demonstrating various features of the robot-kinematics library.

## Installation

Before running the examples, make sure you have the required dependencies installed:

```bash
# Basic dependencies (should already be installed)
pip install numpy scipy matplotlib pyyaml urdfpy

# For Pinocchio backend examples
conda install pinocchio -c conda-forge

# For IK functionality (optional but recommended)
pip install casadi
```

## Available Examples

### 1. URDFPy Backend Example
**File:** `urdfpy_backend_example.py`

Demonstrates the URDFPy backend for robot kinematics:
- Loading a robot from URDF
- Computing forward kinematics (FK)
- Computing Jacobian matrices
- Working with joint configurations

```bash
python urdfpy_backend_example.py
```

### 2. Pinocchio Backend Example
**File:** `pinocchio_backend_example.py`

Demonstrates the Pinocchio backend for robot kinematics:
- Loading a robot from URDF
- Computing forward kinematics (FK)
- Computing FK for all frames
- Finding frames by name
- Locking specific joints

```bash
python pinocchio_backend_example.py
```

### 3. Pinocchio IK Example (NEW)
**File:** `pinocchio_ik_example.py`

Demonstrates inverse kinematics (IK) using the Pinocchio backend with CasADi optimization:
- Initializing the Pinocchio backend
- Computing forward kinematics to get a target pose
- Solving inverse kinematics to reach the target
- Verifying the IK solution accuracy
- Error analysis (position and orientation)

**Requirements:**
- Pinocchio: `conda install pinocchio -c conda-forge`
- CasADi: `pip install casadi`

```bash
python pinocchio_ik_example.py
```

**Features:**
- CasADi-based constrained optimization
- Automatic joint limit enforcement
- Convergence status reporting
- Solution verification with error metrics

### 4. A2D Arm IK Example (NEW)
**File:** `a2d_arm_ik_example.py`

Advanced example showing how to create specialized IK solvers for specific robots (A2D dual-arm robot):
- Creating robot-specific IK solver classes
- Handling body joint configurations
- Locking unnecessary joints for focused IK
- Forward and inverse kinematics integration
- Converting between transformation matrices and poses

**Requirements:**
- Pinocchio: `conda install pinocchio -c conda-forge`
- CasADi: `pip install casadi`
- A2D robot URDF (adjust paths in the script)

```bash
python a2d_arm_ik_example.py
```

**Key Concepts:**
- Wrapping the generic backend for specialized robots
- Joint locking strategy for dual-arm robots
- Utility functions for pose representations
- Model information inspection

## Quick Start

The simplest way to get started is with the basic IK example:

```python
from robot_kinematics.backends.pinocchio_backend import PinocchioKinematicsBackend
from robot_kinematics.frames.transforms import Pose
import numpy as np

# Initialize backend
backend = PinocchioKinematicsBackend(
    urdf_path="path/to/robot.urdf",
    base_link="base_link",
    ee_link="end_effector",
)

# Define target pose
target = Pose(
    position=np.array([0.5, 0.2, 0.3]),
    quaternion=np.array([0, 0, 0, 1])  # [qx, qy, qz, qw]
)

# Solve IK
solution = backend.ik(target_pose=target)
print(f"Joint solution: {solution}")
```

## Example Modifications

### Adapting Examples to Your Robot

To use these examples with your own robot:

1. **Update URDF paths:**
   ```python
   urdf_path = "path/to/your/robot.urdf"
   package_dirs = ["path/to/your/meshes"]
   ```

2. **Set correct link names:**
   ```python
   base_link = "your_base_link"
   ee_link = "your_end_effector_link"
   ```

3. **Configure joints to lock (optional):**
   ```python
   joints_to_lock = [
       "joint_to_lock_1",
       "joint_to_lock_2",
       # ... more joints
   ]
   ```

### Custom IK Solver Options

You can customize the IK solver behavior:

```python
# Custom IPOPT options
custom_options = {
    'ipopt': {
        'print_level': 5,      # Verbose output
        'max_iter': 500,       # More iterations
        'tol': 1e-6,          # Tighter tolerance
    }
}

# Solve with custom options
solution = backend.ik(
    target_pose=target,
    solver_options=custom_options,
    max_iterations=500,
    tolerance=1e-6
)
```

### Getting Detailed Results

For debugging or verification:

```python
# Get convergence status and achieved pose
solution_q, success, achieved_pose = backend.ik(
    target_pose=target,
    return_success=True  # Enable detailed results
)

print(f"Converged: {success}")
print(f"Target: {target.position}")
print(f"Achieved: {achieved_pose.position}")
print(f"Error: {np.linalg.norm(target.position - achieved_pose.position)}")
```

## Troubleshooting

### Common Issues

1. **"Pinocchio is not installed"**
   ```bash
   conda install pinocchio -c conda-forge
   ```

2. **"CasADi is not installed"**
   ```bash
   pip install casadi
   ```

3. **"End-effector link not found"**
   - Check your URDF file for the correct link name
   - Use `backend.list_links()` to see available links

4. **IK doesn't converge**
   - Increase `max_iterations` (e.g., 500 or 1000)
   - Relax `tolerance` (e.g., 1e-3 instead of 1e-4)
   - Check if target pose is within robot workspace
   - Provide better initial guess using FK from nearby configuration

5. **"meshes not found" errors**
   - Make sure `package_dirs` points to directory containing mesh files
   - Check that mesh paths in URDF are relative to package directory

### Performance Tips

- **Reuse backend object**: Don't recreate for each IK call
- **Use warm starts**: Pass previous solution as `initial_joint_positions`
- **Lock unnecessary joints**: Reduces problem dimensionality
- **Adjust tolerance**: Balance accuracy vs speed

## Documentation

For more detailed information:

- **IK Implementation Details:** See `docs/IK_IMPLEMENTATION.md`
- **Implementation Summary:** See `IK_IMPLEMENTATION_SUMMARY.md` in the root directory
- **API Documentation:** See docstrings in the source code

## Additional Resources

### Robot Models

The examples assume you have robot URDF files. Common sources:

- Your own robot models
- ROS robot descriptions
- Public repositories (e.g., robot_descriptions.py)

### Learning More

- **Pinocchio:** https://stack-of-tasks.github.io/pinocchio/
- **CasADi:** https://web.casadi.org/
- **URDF Format:** http://wiki.ros.org/urdf/XML
- **SE(3) Geometry:** Murray et al., "A Mathematical Introduction to Robotic Manipulation"

## Contributing

To contribute new examples:

1. Follow the existing example structure
2. Include clear comments and docstrings
3. Add entry to this README
4. Test with at least one robot model

## License

These examples are part of the robot-kinematics package and follow the same MIT license.
