# Backend Inheritance Architecture Guide

This document explains how the kinematics backend inheritance works and how subclasses inherit shared URDF inspector methods.

## Architecture Overview

```
BaseKinematicsBackend (Abstract Base Class)
├── _urdf_inspector (attribute)
├── _get_urdf_inspector() (shared method)
├── list_links() (shared method)
├── list_joints() (shared method)
├── fk() (abstract method)
├── ik() (abstract method)
└── fk_all_frames() (abstract method)

URDFPyKinematicsBackend (Concrete Implementation)
└── inherits all shared methods from BaseKinematicsBackend

PinocchioKinematicsBackend (Concrete Implementation)
└── inherits all shared methods from BaseKinematicsBackend
```

## How Inheritance Works

### 1. Base Class: `BaseKinematicsBackend`

The base class provides:
- **Shared utility methods** that use the URDF inspector
- **Abstract methods** that subclasses must implement

```python
class BaseKinematicsBackend(ABC):
    def __init__(self):
        # Initialize inspector as None - subclasses will set this
        self._urdf_inspector = None
    
    def _get_urdf_inspector(self) -> Optional[SubchainURDFInspector]:
        """Returns the cached URDF inspector."""
        return self._urdf_inspector
    
    def list_links(self) -> List[str]:
        """Uses inspector to get all links in kinematic chain."""
        inspector = self._get_urdf_inspector()
        if inspector is not None:
            return inspector.list_links()
        return []
    
    def list_joints(self, movable_only: bool = True) -> List[str]:
        """Uses inspector to get all joints in kinematic chain."""
        inspector = self._get_urdf_inspector()
        if inspector is not None:
            return inspector.get_joint_names(movable_only=movable_only)
        return []
    
    @abstractmethod
    def fk(self, joint_positions, target_link=None) -> Pose:
        """Subclasses must implement forward kinematics."""
        ...
```

### 2. Subclass: `URDFPyKinematicsBackend`

The subclass inherits shared methods by:
1. Initializing `self._urdf_inspector` in its `__init__`
2. Implementing abstract methods

```python
class URDFPyKinematicsBackend(BaseKinematicsBackend):
    def __init__(self, urdf_path, base_link, ee_link, ...):
        # Step 1: Set required attributes
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        
        # Step 2: Initialize URDF inspector for shared methods
        self._urdf_inspector = SubchainURDFInspector(
            urdf_path, base_link, ee_link
        )
        
        # Step 3: Now can use inherited methods!
        joint_names = self.list_joints(movable_only=True)  # Works!
        link_names = self.list_links()  # Works!
        
        # Step 4: Backend-specific initialization
        self.robot = URDF.load(urdf_path)
        # ... rest of initialization
    
    def fk(self, joint_positions, target_link=None) -> Pose:
        # Backend-specific implementation
        ...
```

### 3. Subclass: `PinocchioKinematicsBackend`

Same pattern as URDFPy:

```python
class PinocchioKinematicsBackend(BaseKinematicsBackend):
    def __init__(self, urdf_path, base_link, ee_link, ...):
        # Step 1: Set required attributes
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        
        # Step 2: Initialize URDF inspector for shared methods
        self._urdf_inspector = SubchainURDFInspector(
            urdf_path, base_link, ee_link
        )
        
        # Step 3: Use inherited methods
        if joint_names is None:
            joint_names = self.list_joints(movable_only=True)
        
        # Step 4: Backend-specific initialization
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
        # ... rest of initialization
    
    def fk(self, joint_positions, target_link=None) -> Pose:
        # Backend-specific implementation using Pinocchio
        ...
```

## Method Execution Flow

When you call an inherited method, here's what happens:

```python
# Create a backend instance
backend = URDFPyKinematicsBackend(
    urdf_path="robot.urdf",
    base_link="base_link",
    ee_link="ee_link"
)

# Call inherited method
joints = backend.list_joints()

# Execution flow:
# 1. Python looks for list_joints() in URDFPyKinematicsBackend
# 2. Not found, so looks in parent: BaseKinematicsBackend
# 3. Found! Calls BaseKinematicsBackend.list_joints()
# 4. That method calls self._get_urdf_inspector()
# 5. Returns self._urdf_inspector (initialized in URDFPyKinematicsBackend.__init__)
# 6. Calls inspector.get_joint_names(movable_only=True)
# 7. Returns the joint list
```

## Key Points

### ✅ What Subclasses MUST Do

1. **Initialize `self._urdf_inspector`** in their `__init__`:
   ```python
   self._urdf_inspector = SubchainURDFInspector(urdf_path, base_link, ee_link)
   ```

2. **Set required attributes** before using inherited methods:
   ```python
   self.urdf_path = urdf_path
   self.base_link = base_link
   self.ee_link = ee_link
   ```

3. **Implement all abstract methods** (fk, ik, fk_all_frames)

### ✅ What Subclasses GET for Free

Once `_urdf_inspector` is initialized, subclasses automatically inherit:
- `list_links()` - Get all links in kinematic chain
- `list_joints(movable_only=True)` - Get joints in chain
- `_get_urdf_inspector()` - Access to the inspector

### ❌ What NOT to Do

1. **Don't call `super().__init__()`** with the old wrapper arguments:
   ```python
   # ❌ WRONG - this was the old wrapper pattern
   super().__init__(backend="urdfpy", extra={...})
   ```

2. **Don't forget to initialize `_urdf_inspector`**:
   ```python
   # ❌ WRONG - will cause AttributeError
   def __init__(self, ...):
       self.urdf_path = urdf_path
       # Missing: self._urdf_inspector = ...
       joints = self.list_joints()  # ERROR!
   ```

## Example: Adding a New Backend

To create a new backend (e.g., `MyCustomBackend`):

```python
from robot_kinematics.backends.base_kinematics_backend import BaseKinematicsBackend
from robot_kinematics.urdf.inspector import SubchainURDFInspector
from robot_kinematics.frames import Pose

class MyCustomBackend(BaseKinematicsBackend):
    def __init__(self, urdf_path, base_link, ee_link, **kwargs):
        # 1. Store required attributes
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        
        # 2. Initialize URDF inspector (enables list_links/list_joints)
        self._urdf_inspector = SubchainURDFInspector(
            urdf_path, base_link, ee_link
        )
        
        # 3. Use inherited methods to get joint/link info
        self.joint_names = self.list_joints(movable_only=True)
        self.link_names = self.list_links()
        self.n_dof = len(self.joint_names)
        
        # 4. Initialize your custom backend library
        self.my_robot = MyLibrary.load(urdf_path)
        # ... rest of your initialization
    
    def fk(self, joint_positions, target_link=None):
        """Implement using your library."""
        if target_link is None:
            target_link = self.ee_link
        # Your FK implementation here
        return Pose(...)
    
    def ik(self, target_pose, initial_joint_positions=None, **kwargs):
        """Implement using your library."""
        # Your IK implementation here
        return np.array([...])
    
    def fk_all_frames(self, joint_positions):
        """Implement using your library."""
        # Your FK for all frames implementation here
        return {"link1": Pose(...), "link2": Pose(...), ...}
```

## Benefits of This Architecture

1. **Code Reuse**: URDF inspection logic written once, used by all backends
2. **Consistency**: All backends provide the same interface for listing joints/links
3. **Flexibility**: Subclasses can override shared methods if needed
4. **Clean Separation**: Backend-specific code stays in backend classes
5. **Easy to Extend**: Adding new backends is straightforward

## Testing the Inheritance

To verify inheritance is working:

```python
# Test URDFPy backend
backend = URDFPyKinematicsBackend(
    urdf_path="robot.urdf",
    base_link="base",
    ee_link="ee"
)

# These should work without errors
print(backend.list_joints())  # Uses inherited method
print(backend.list_links())   # Uses inherited method

# Test Pinocchio backend
backend2 = PinocchioKinematicsBackend(
    urdf_path="robot.urdf",
    base_link="base",
    ee_link="ee"
)

# These should also work
print(backend2.list_joints())  # Same inherited method
print(backend2.list_links())   # Same inherited method
```

## Troubleshooting

### Error: `'URDFPyKinematicsBackend' object has no attribute '_urdf_inspector'`

**Cause**: The `_urdf_inspector` was not initialized in the subclass `__init__`.

**Solution**: Add this line in your `__init__`:
```python
self._urdf_inspector = SubchainURDFInspector(urdf_path, base_link, ee_link)
```

### Error: Inspector returns empty lists

**Cause**: The `urdf_path`, `base_link`, or `ee_link` attributes are not set before creating the inspector.

**Solution**: Set these attributes first:
```python
self.urdf_path = urdf_path
self.base_link = base_link
self.ee_link = ee_link
self._urdf_inspector = SubchainURDFInspector(urdf_path, base_link, ee_link)
```
