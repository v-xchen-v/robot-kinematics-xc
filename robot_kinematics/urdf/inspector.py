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