try:
    from ..compat import urdfpy_compat
except ImportError:
    import os.path as osp
    PROJECT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import sys
    sys.path.insert(0, PROJECT_DIR)
    from robot_kinematics.compat import urdfpy_compat

from urdfpy import URDF
from typing import List, Tuple, Optional
from collections import deque
from collections import defaultdict


# --- Full URDF Inspector ---
class FullURDFInspector:
    """
    Inspect the full URDF: all joints, links, and tree structure.
    """
    def __init__(self, urdf_path: str):
        self.robot = URDF.load(urdf_path)
        self.urdf_path = urdf_path
        print(f"Loaded URDF: {self.robot.name}")
        # Build link connectivity maps
        self.child_map = defaultdict(list)
        self.parent_map = {}
        for j in self.robot.joints:
            self.child_map[j.parent].append((j.name, j.child))
            self.parent_map[j.child] = (j.name, j.parent)

    def list_joints(self, movable_only: bool = True, verbose=False):
        joints = self.robot.joints
        if movable_only:
            joints = [j for j in joints if j.joint_type != 'fixed']
        if verbose:
            print(f"\nLoaded URDF: {self.robot.name}")
            print(f"Total joints: {len(self.robot.joints)} | Movable: {len(joints)}")
            print("-" * 60)
            for j in joints:
                print(f"{j.name:<20} type={j.joint_type:<8} parent={j.parent} → child={j.child} "
                      f"limit={getattr(j.limit, 'lower', None)}~{getattr(j.limit, 'upper', None)}")
        return joints

    def list_links(self, verbose=False):
        links = self.robot.links
        if verbose:
            print(f"\nLinks in robot '{self.robot.name}':")
            for link in links:
                print(f"  {link.name}")
        return links

    def get_joint_names(self, movable_only=True):
        return [
            j.name for j in self.robot.joints
            if (not movable_only or j.joint_type != 'fixed')
        ]

    def list_joint_limits(self, movable_only: bool = True, verbose: bool = False):
        """
        List joint limits for all joints or movable joints only.
        
        Args:
            movable_only: If True, only include movable joints (non-fixed)
            verbose: If True, print detailed information
            
        Returns:
            List of tuples (joint_name, joint_type, lower_limit, upper_limit)
        """
        joints = self.robot.joints
        if movable_only:
            joints = [j for j in joints if j.joint_type != 'fixed']
        
        joint_limits = []
        for j in joints:
            lower = getattr(j.limit, 'lower', None) if j.limit else None
            upper = getattr(j.limit, 'upper', None) if j.limit else None
            joint_limits.append((j.name, j.joint_type, lower, upper))
        
        if verbose:
            print(f"\nJoint limits for robot '{self.robot.name}':")
            print(f"{'Joint Name':<25} {'Type':<12} {'Lower Limit':<15} {'Upper Limit':<15}")
            print("-" * 70)
            for name, jtype, lower, upper in joint_limits:
                lower_str = f"{lower:.3f}" if lower is not None else "None"
                upper_str = f"{upper:.3f}" if upper is not None else "None"
                print(f"{name:<25} {jtype:<12} {lower_str:<15} {upper_str:<15}")
        
        return joint_limits
    
    def print_tree(self):
        print(f"\nRobot link hierarchy ({self.robot.name}):")
        def recurse(link, indent=0):
            children = self.child_map.get(link, [])
            print("  " * indent + f"- {link}")
            for joint_name, child_link in children:
                print("  " * (indent + 1) + f"↳ [{joint_name}] → {child_link}")
                recurse(child_link, indent + 2)
        base = self.robot.base_link.name
        recurse(base)

    def find_connecting_joint(self, parent_link: str, child_link: str) -> Optional[str]:
        for joint_name, c_link in self.child_map.get(parent_link, []):
            if c_link == child_link:
                return joint_name
        return None

    def get_joint_chain(self, link_start: str, link_end: str, movable_only: bool = True) -> List[Tuple[str, str, str]]:
        visited = set()
        queue = deque([link_start])
        parent_track = {}
        found = False
        while queue:
            cur = queue.popleft()
            if cur == link_end:
                found = True
                break
            for jname, child in self.child_map.get(cur, []):
                if child not in visited:
                    visited.add(child)
                    parent_track[child] = (cur, jname)
                    queue.append(child)
            if cur in self.parent_map:
                jname, parent = self.parent_map[cur]
                if parent not in visited:
                    visited.add(parent)
                    parent_track[parent] = (cur, jname)
                    queue.append(parent)
        if not found:
            raise ValueError(f"No path between '{link_start}' and '{link_end}'")
        path_links = [link_end]
        while path_links[-1] != link_start:
            prev, jname = parent_track[path_links[-1]]
            path_links.append(prev)
        path_links.reverse()
        chain = []
        for a, b in zip(path_links[:-1], path_links[1:]):
            jname = self.find_connecting_joint(a, b) or self.find_connecting_joint(b, a)
            if movable_only:
                j = self.robot.joint_map[jname]
                if j.joint_type == 'fixed':
                    continue
            chain.append((jname, a, b))
        return chain

    def get_link_chain(self, link_start: str, link_end: str) -> List[str]:
        chain = self.get_joint_chain(link_start, link_end)
        links = [link_start]
        for _, _, child_link in chain:
            links.append(child_link)
        return links

    def print_joint_chain(self, link_start: str, link_end: str, movable_only: bool = True):
        chain = self.get_joint_chain(link_start, link_end, movable_only=movable_only)
        print(f"\nJoint chain from '{link_start}' to '{link_end}':")
        for jname, a, b in chain:
            j = self.robot.joint_map[jname]
            print(f"  {a} --[{jname} ({j.joint_type})]--> {b}")

    def list_excluded_joints(
        self, 
        base_link: str, 
        ee_link: str, 
        movable_only: bool = True, 
        active_joints: Optional[List[str]] = None,
        verbose: bool = False
    ) -> List[str]:
        """
        List joints that are NOT part of the chain between base_link and ee_link.
        
        Args:
            base_link: The base link name
            ee_link: The end effector link name
            movable_only: If True, only consider movable joints (non-fixed)
            verbose: If True, print detailed information
            
        Returns:
            List of joint names that are excluded from the chain
        """
        # Get joints in the chain
        chain_joints = set()
        try:
            joint_chain = self.get_joint_chain(base_link, ee_link, movable_only=movable_only)
            chain_joints = {jname for jname, _, _ in joint_chain}
        except ValueError:
            # If no path exists, all joints are excluded
            pass
        
        # Get all joints in the robot
        all_joints = self.robot.joints
        if movable_only:
            all_joints = [j for j in all_joints if j.joint_type != 'fixed']
        
        # list non-active joints that are in chain
        active_chain_joints = []
        if active_joints is not None:
            for j in chain_joints:
                if j in active_joints:
                    active_chain_joints.append(j)
                    if verbose:
                        print(f"  {j:<20} (inactive)")
                        
        # Find excluded joints
        excluded_joints = [j.name for j in all_joints if j.name not in active_chain_joints]
        
        if verbose:
            print(f"\nExcluded joints (not in chain from '{base_link}' to '{ee_link}'):")
            for joint_name in excluded_joints:
                joint = self.robot.joint_map[joint_name]
                print(f"  {joint_name:<20} type={joint.joint_type:<8} parent={joint.parent} → child={joint.child}")
            print(f"Total excluded: {len(excluded_joints)} out of {len([j for j in self.robot.joints if not movable_only or j.joint_type != 'fixed'])}")
        
        return excluded_joints

# --- Subchain Inspector ---
class SubchainURDFInspector:
    """
    Inspect a subchain between a specified base_link and end_link.
    """
    def __init__(self, urdf_path: str, base_link: str, end_link: str):
        self.full_inspector = FullURDFInspector(urdf_path)
        self.base_link = base_link
        self.end_link = end_link
        self.robot = self.full_inspector.robot
        self.joint_chain = self.full_inspector.get_joint_chain(base_link, end_link)
        self.link_chain = self.full_inspector.get_link_chain(base_link, end_link)

    def list_joints(self, movable_only: bool = True, verbose=False):
        joints = []
        for jname, a, b in self.full_inspector.get_joint_chain(self.base_link, self.end_link, movable_only=movable_only):
            joints.append(self.robot.joint_map[jname])
        if verbose:
            print(f"\nJoints in subchain '{self.base_link}' to '{self.end_link}':")
            for j in joints:
                print(f"{j.name:<20} type={j.joint_type:<8} parent={j.parent} → child={j.child} "
                      f"limit={getattr(j.limit, 'lower', None)}~{getattr(j.limit, 'upper', None)}")
        return joints

    def list_links(self, verbose=False):
        links = self.link_chain
        if verbose:
            print(f"\nLinks in subchain '{self.base_link}' to '{self.end_link}':")
            for link in links:
                print(f"  {link}")
        return links

    def get_joint_names(self, movable_only=True):
        return [j.name for j in self.list_joints(movable_only=movable_only)]

    def list_joint_limits(self, movable_only: bool = True, verbose: bool = False):
        """
        List joint limits for joints in the subchain.
        
        Args:
            movable_only: If True, only include movable joints (non-fixed)
            verbose: If True, print detailed information
            
        Returns:
            List of tuples (joint_name, joint_type, lower_limit, upper_limit)
        """
        joints = self.list_joints(movable_only=movable_only)
        joint_limits = []
        
        for j in joints:
            lower = getattr(j.limit, 'lower', None) if j.limit else None
            upper = getattr(j.limit, 'upper', None) if j.limit else None
            joint_limits.append((j.name, j.joint_type, lower, upper))
        
        if verbose:
            print(f"\nJoint limits for subchain '{self.base_link}' to '{self.end_link}':")
            print(f"{'Joint Name':<25} {'Type':<12} {'Lower Limit':<15} {'Upper Limit':<15}")
            print("-" * 70)
            for name, jtype, lower, upper in joint_limits:
                lower_str = f"{lower:.3f}" if lower is not None else "None"
                upper_str = f"{upper:.3f}" if upper is not None else "None"
                print(f"{name:<25} {jtype:<12} {lower_str:<15} {upper_str:<15}")
        
        return joint_limits

    def list_excluded_joints(self, verbose: bool = False) -> List[str]:
        """
        List joints that are NOT part of the subchain between base_link and end_link.
        
        Args:
            verbose: If True, print detailed information
            
        Returns:
            List of joint names that are excluded from the subchain
        """
        return self.full_inspector.list_excluded_joints(self.base_link, self.end_link, verbose=verbose)

    def print_joint_chain(self, movable_only: bool = True):
        chain = self.full_inspector.get_joint_chain(self.base_link, self.end_link, movable_only=movable_only)
        print(f"\nJoint chain from '{self.base_link}' to '{self.end_link}':")
        for jname, a, b in chain:
            j = self.robot.joint_map[jname]
            print(f"  {a} --[{jname} ({j.joint_type})]--> {b}")
        
if __name__ == "__main__":
    urdf_path = "robots/g1/G1_120s/urdf/G1_120s.urdf"
    full_inspector = FullURDFInspector(urdf_path)
    base_link_name = full_inspector.robot.base_link.name
    ee_name = "gripper_l_center_link"
    print(full_inspector.get_joint_chain(base_link_name, ee_name, movable_only=True))
    full_inspector.print_joint_chain(base_link_name, ee_name, movable_only=True)
    # Subchain example
    sub_inspector = SubchainURDFInspector(urdf_path, base_link_name, ee_name)
    sub_inspector.print_joint_chain(movable_only=True)