import os
import time

import numpy as np

from misc.urdf_editor import UrdfEditor
from gripper_module.gripper_base import GripperBase
from gripper_module.articulated_gripper import fetch_gripper, gripper_names

class ArticulatedGripper(GripperBase):
    
    def __init__(self,
                 bullet_client,
                 home_position,
                 gripper_size,
                 gripper_name,
                 ):
        super().__init__(home_position, gripper_size)
        
        self._bullet_client = bullet_client
        
        # parse gripper name
        if gripper_name is None:
            self._gripper_name = np.random.choice(gripper_names)
        else:
            self._gripper_name = gripper_name
        
        self.load_gripper()
        
        # define force and speed (movement of mount)
        self._force = 10000
        self._speed = 0.005
        
        self.fix_joints(range(self._bullet_client.getNumJoints(self._body_id)))
        
        self._pose_joints = list(range(6))

    def load_gripper(self):
        self._gripper = fetch_gripper(self._gripper_name)(self._bullet_client, self._gripper_size)
        gripper_body_id = self._gripper.load(self._home_position)
        
        # load mount
        mount_urdf = 'assets/gripper/mount.urdf'
        mount_body_id = self._bullet_client.loadURDF(
            mount_urdf,
            basePosition=self._home_position,
            useFixedBase=True
        )

        # combine mount and gripper by a joint
        ed_mount = UrdfEditor()
        ed_mount.initializeFromBulletBody(mount_body_id, self._bullet_client._client)
        ed_gripper = UrdfEditor()
        ed_gripper.initializeFromBulletBody(gripper_body_id, self._bullet_client._client)

        self._gripper_parent_index = 6
        newjoint = ed_mount.joinUrdf(
            childEditor=ed_gripper,
            parentLinkIndex=self._gripper_parent_index,
            jointPivotXYZInParent=self._gripper.get_pos_offset(),
            jointPivotRPYInParent=self._bullet_client.getEulerFromQuaternion(self._gripper.get_orn_offset()),
            jointPivotXYZInChild=[0, 0, 0],
            jointPivotRPYInChild=[0, 0, 0],
            parentPhysicsClientId=self._bullet_client._client,
            childPhysicsClientId=self._bullet_client._client
        )
        newjoint.joint_type = self._bullet_client.JOINT_FIXED
        newjoint.joint_name = "joint_mount_gripper"
        urdfname = f".tmp_combined_{self._gripper_name}_{self._gripper_size:.4f}_{np.random.random():.10f}_{time.time():.10f}.urdf"
        ed_mount.saveUrdf(urdfname)
        # remove mount and gripper bodies
        self._bullet_client.removeBody(mount_body_id)
        self._bullet_client.removeBody(gripper_body_id)

        self._body_id = self._bullet_client.loadURDF(
            urdfname,
            useFixedBase=True,
            basePosition=self._home_position,
            baseOrientation=self._bullet_client.getQuaternionFromEuler([0, 0, 0])
        )
        
        # remove the combined URDF
        os.remove(urdfname)
        
        # configure the gripper (e.g. friction)
        self._gripper.configure(self._body_id, self._gripper_parent_index+1)
        
    def fix_joints(self, joint_ids):
        """fix the joints at current position
        Args:
            joint_ids: list of int
                the joint id of joints to fix.
        """
        current_states = np.array([self._bullet_client.getJointState(self._body_id, joint_id)[0] for joint_id in joint_ids])
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=current_states,
            forces=[self._force] * len(joint_ids),
            # positionGains=[self._speed] * len(joint_ids)
        )
        
        
    def get_joint_limit(self):
        return self._gripper.get_joint_limit()
    
    def step_pose(self, pose):
        self.move([pose[0], pose[1], self._home_position[2]], pose[3])
        self.move([pose[0], pose[1], pose[2]], pose[3])
    
    def step_joints(self, target_states):
        self._gripper.step_joints(target_states)
    
    def reset(self):
        self.move(self._home_position, 0)
        self._gripper.reset()
    
    def up(self):
        pose = [self._bullet_client.getJointState(self._body_id, id)[0] for id in self._pose_joints]
        self.move([pose[0], pose[1], self._home_position[2]], pose[5])
    
    def get_joint_states(self):
        return self._gripper.get_joint_states()
    
    def move(self, target_position, rotation_angle, stop_at_contact=False):
        target_position = np.array(target_position) - np.array(self._home_position)
        target_states = [target_position[0], target_position[1], target_position[2], 0, 0, rotation_angle%(2*np.pi)]
        
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            self._pose_joints,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_states,
            forces=[self._force] * len(self._pose_joints),
            positionGains=[self._speed] * len(self._pose_joints)
        )
        
        for i in range(240 * 6):
            current_states = np.array([self._bullet_client.getJointState(self._body_id, id)[0] for id in self._pose_joints])
            states_diff = np.abs(target_states - current_states)
            # stop moving gripper if gripper collide with other objects
            if stop_at_contact:
                is_in_contact = False
                points = self._bullet_client.getContactPoints(bodyA=self._body_id)
                if len(points) > 0:
                    for p in points:
                        if p[9] > 0:
                            is_in_contact = True
                            break
                if is_in_contact:
                    break
            if np.all(states_diff < 1e-4):
                break
            # self._gripper.step_constraints(self._body_id, self._gripper_parent_index+1)
            self._bullet_client.stepSimulation()
        self.fix_joints(self._pose_joints)
        
        
    def get_id(self):
        return self._body_id