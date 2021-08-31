import numpy as np
import time


class GripperBarrettHand():
    def __init__(self, bullet_client, gripper_size, palm_joint=None, palm_joint_another=None):
        """ Initialization of barrett hand
        specific args for barrett hand:
            - gripper_size: global scaling of the gripper when loading URDF
        """

        self._bullet_client = bullet_client
        self._gripper_size = gripper_size
        # self._finger_rotation1 = np.pi * palm_joint
        # self._finger_rotation2 = np.pi * palm_joint_another if palm_joint_another is not None else np.pi * palm_joint
        self._pos_offset = np.array([0, 0, 0.185 * self._gripper_size]) # offset from base to center of grasping
        # self._pos_offset = np.array([0, 0, 0.181 * self._gripper_size]) # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi / 2])
        
        # define force and speed (grasping)
        self._force = 100
        self._grasp_speed = 2

        self._finger_force = 30
        self._finger_grasp_speed = 0.001

        # define driver joint; the follower joints need to satisfy constraints when grasping
        finger1_joint_ids = [1, 2] # thumb
        finger2_joint_ids = [4, 5] # index finger
        finger3_joint_ids = [7, 8] # middle finger
        self._finger_joint_ids = finger1_joint_ids+finger2_joint_ids+finger3_joint_ids

        self._control_joint_ids = [1, 2, 3, 4, 5, 6, 7, 8]

        self._palm_joint_ids = [3, 6]
        self._joint_lower = 1
        self._joint_upper = 1.6
        

    def load(self, basePosition):
        gripper_urdf = "assets/gripper/barrett_hand/model.urdf"
        body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=basePosition
        )
        return body_id

    def configure(self, mount_gripper_id, n_links_before):
        self._body_id = mount_gripper_id
        self._n_links_before = n_links_before
        # Set friction coefficients for gripper fingers
        for i in range(n_links_before, self._bullet_client.getNumJoints(self._body_id)):
            self._bullet_client.changeDynamics(self._body_id,i,lateralFriction=1.0,spinningFriction=1.0,rollingFriction=0.0001,frictionAnchor=True)
        # set joint limit
        lower = [self._bullet_client.getJointInfo(self._body_id, id+self._n_links_before)[8] for id in self._control_joint_ids]
        upper = [self._bullet_client.getJointInfo(self._body_id, id+self._n_links_before)[9] for id in self._control_joint_ids]
        for i in self._palm_joint_ids:
            upper[i-1]=0.5*np.pi # palm joints do not surpass .5pi
        self._joint_limit_upper = np.array(upper)
        self._joint_limit_lower = np.array(lower)

    def step_constraints(self):
        # keep finger2 and finger3
        current_states = np.array([self._bullet_client.getJointState(self._body_id, id+self._n_links_before)[0] for id in self._palm_joint_ids])
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            self._palm_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=current_states,
            forces=[self._force] * len(self._palm_joint_ids),
            # positionGains=[self._speed] * len(joint_ids)
        )


    # def open(self, mount_gripper_id, n_joints_before, open_scale):
    #     target_pos = open_scale*self._joint_lower + (1-open_scale)*self._joint_upper  # recalculate scale because larger joint position corresponds to smaller open width
    #     self._bullet_client.setJointMotorControl2(
    #         mount_gripper_id,
    #         self._driver_joint_id+n_joints_before,
    #         self._bullet_client.POSITION_CONTROL,
    #         targetPosition=target_pos,
    #         force=self._force
    #     )
    #     for i in range(240 * 2):
    #         pos = self.step_constraints(mount_gripper_id, n_joints_before)
    #         if np.abs(target_pos-pos)<1e-5:
    #             break
    #         self._bullet_client.stepSimulation()

    
    # def close(self, mount_gripper_id, n_joints_before):
    #     self._bullet_client.setJointMotorControl2(
    #         mount_gripper_id,
    #         self._driver_joint_id+n_joints_before,
    #         self._bullet_client.VELOCITY_CONTROL,
    #         targetVelocity=self._grasp_speed,
    #         force=self._force
    #     )
    #     for i in range(240 * 2):
    #         pos = self.step_constraints(mount_gripper_id, n_joints_before)
    #         if pos>self._joint_upper+0.1:
    #             break
    #         self._bullet_client.stepSimulation()

    def get_joint_states(self):
        joint_states = [self._bullet_client.getJointState(self._body_id, id+self._n_links_before)[0] for id in self._control_joint_ids]
        return joint_states
    
    def get_joint_limit(self):
        return np.stack([self._joint_limit_lower, self._joint_limit_upper], axis=0)

    def position_control(self, moving_joint_ids, target_states, static_joint_ids=None, detect_contact=False):
        if detect_contact:
            contact_object_ids = set()
            
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            [id+self._n_links_before for id in moving_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_states,
            # targetVelocities=[0.001] * len(target_states),
            forces=[self._finger_force] * len(moving_joint_ids),
            positionGains=[self._finger_grasp_speed] * len(moving_joint_ids)
        )
        if static_joint_ids:
            static_joint_states = [self._bullet_client.getJointState(self._body_id, id+self._n_links_before)[0] for id in static_joint_ids]
            self._bullet_client.setJointMotorControlArray(
                self._body_id,
                [id+self._n_links_before for id in static_joint_ids],
                self._bullet_client.POSITION_CONTROL,
                targetPositions=static_joint_states,
                # targetVelocities=[0.001] * len(target_states),
                forces=[self._force] * len(static_joint_states),
                # positionGains=[self._finger_grasp_speed] * len(static_joint_states)
            )
        for i in range(int(240 * 40)):
            current_states = np.array([self._bullet_client.getJointState(self._body_id, id+self._n_links_before)[0] for id in moving_joint_ids])
            states_diff = np.abs(target_states - current_states)
            if np.all(states_diff < 1e-4):
                break
            self._bullet_client.stepSimulation()

            if detect_contact:
                points = self._bullet_client.getContactPoints(bodyA=self._body_id)
                if len(points) > 0:
                    for p in points:
                        if p[9] > 0:
                            contact_object_ids.add(p[2])
                            # if i%240==0:
                            #     self._bullet_client.addUserDebugLine(p[5], np.array(p[5])+0.05*np.array(p[7]), lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0.1)
            
        if detect_contact:
            return contact_object_ids
        else:
            return set()
        
    def step_joints(self, target_states, detect_contact=False, move_seq=True):

        # controls joint in control_joint_ids to reach target_states
        # return final joint values

        if move_seq:
            contact_object_ids = set()
            # move palm joints
            target_states_palm = [target_states[i-1] for i in self._palm_joint_ids]
            contacts = self.position_control(self._palm_joint_ids, target_states_palm, static_joint_ids=None, detect_contact=detect_contact)
            contact_object_ids.update(contacts)

            # move finger joints
            target_states_finger = [target_states[i-1] for i in self._finger_joint_ids]
            contacts = self.position_control(self._finger_joint_ids, target_states_finger, static_joint_ids=self._palm_joint_ids, detect_contact=detect_contact)
            contact_object_ids.update(contacts)
        else:
            contact_object_ids = self.position_control(self._control_joint_ids, target_states, static_joint_ids=None, detect_contact=detect_contact)

        if detect_contact:
            return np.array(self.get_joint_states()), contact_object_ids
        else:
            return np.array(self.get_joint_states())

    def reset(self):
        self.step_joints(self._joint_limit_lower)
    
    
    def get_pos_offset(self):
        return self._pos_offset

    
    def get_orn_offset(self):
        return self._orn_offset
    

    # def get_body_id(self):
    #     return self._body_id


    # def get_vis_pts(self, open_scale):
    #     k = 0.025 + 0.049 * np.sin(2*open_scale - 0.8455)
    #     m = 0.025
    #     return self._gripper_size * np.array([
    #         [-k * np.cos(self._finger_rotation1), -m - k * np.sin(self._finger_rotation1)],
    #         [-k * np.cos(self._finger_rotation2), m + k * np.sin(self._finger_rotation2)],
    #         [k, 0]
    #     ])