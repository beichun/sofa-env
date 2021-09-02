import numpy as np

import Sofa
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from splib3.loaders import loadPointListFromFile

from gripper_module.gripper_base import GripperBase


class CableGripper(GripperBase):
    
    def __init__(self,
                 home_position,
                 gripper_size,
                 gripper_name,
                #  n_fingers=3,
                #  long_finger=True,
                 poissonRatio=0.3,
                 youngModulus=10000,
                 gripper_root=None,
                 stepSim=None,
                 unit_scale=1000):
        super().__init__(home_position, gripper_size)
        
        self.parse_name(gripper_name)
        
        self._unit_scale = unit_scale
        if gripper_root is None:
            self._root = Sofa.Core.Node("Gripper")
        else:
            self._root = gripper_root
        self.stepSimulation = stepSim
        self._gripper_category = 'cablegripper'
        
        if self._long_finger:
            self._position_offset = np.array([0, 0, 105.])*self._gripper_size
        else:
            # self._gripper_size *= 2
            self._position_offset = np.array([0, 0, 70.])*self._gripper_size
            
        # physical property
        self._poissonRatio = poissonRatio
        self._youngModulus = youngModulus
        
        # load fingers
        self._fingers = []
        self.load_gripper()
        
        self._current_rot = 0
        self._current_pos = np.array([0, 0, self._home_position[2]])
        self._constraint_limit = 25
        self._joint_dof = self._n_fingers
        self._joint_limit_upper = self._constraint_limit * np.ones(self._n_fingers)
        self._joint_limit_lower = np.zeros(self._n_fingers)
    
    def parse_name(self, gripper_name):
        self._gripper_name = gripper_name
        gripper_char = self._gripper_name.split('_')
        n_fingers = int(gripper_char[0][0])
        if n_fingers in [2, 3, 4]:
            self._n_fingers = n_fingers
        else:
            raise NotImplementedError(f'{n_fingers}-finger cable gripper not implemented.')
        if gripper_char[1]=='long':
            self._long_finger = True
        elif gripper_char[1]=='short':
            self._long_finger = False
        else:
            raise NotImplementedError(f'Unknown gripper finger type: {gripper_char[1]}.')

    def load_gripper(self):
        if self._long_finger:
            finger_vol = "assets/mesh/finger.vtk"
            finger_mesh = "assets/mesh/finger.stl"
            cable_file = "assets/mesh/cable.json"
        else:
            finger_mesh = 'assets/mesh/finger_short.stl'
            finger_vol = "assets/mesh/finger_short.msh"
            cable_file = 'assets/mesh/cable_short.json'
        
        if self._n_fingers==3:
            rotation=[90, -115, 0]
            translation = np.array([10.0, 10.0, 0.0])
            fixingBox = np.array([-20, -15, -10, 20, 20, 10])
            pullPointLocation=np.array([3, 8, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger1', finger_vol, finger_mesh, cable_file)
            )

            rotation=[-90, -65, 0]
            translation=np.array([-10.0, -23.0, 0.0])
            fixingBox = np.array([-20, -30, -10, 20, 0, 0])
            pullPointLocation=np.array([3, -25, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger2', finger_vol, finger_mesh, cable_file)
            )

            rotation=[-90, -65, 0]
            translation=np.array([-10.0, 13.0, 0.0])
            fixingBox = np.array([-20, 5, -10, 20, 35, 10])
            pullPointLocation=np.array([3, 15, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger3', finger_vol, finger_mesh, cable_file)
            )
        
        elif self._n_fingers==2:
            rotation=[90, -115, 0]
            translation = np.array([10.0, 10.0, 0.0])
            fixingBox = np.array([-20, -15, -10, 20, 20, 10])
            pullPointLocation=np.array([3, 8, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger1', finger_vol, finger_mesh, cable_file)
            )
            
            rotation=[-90, -65, 0]
            translation=np.array([-10.0, -5.0, 0.0])
            fixingBox = np.array([-20, -150, -10, 0, 20, 10])
            pullPointLocation=np.array([17, 0, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger2', finger_vol, finger_mesh, cable_file)
            )
        elif self._n_fingers==4:
            finger_gap_y = 20
            rotation=[90, -115, 0]
            translation = np.array([10.0, 10.0-finger_gap_y, 0.0])
            fixingBox = np.array([-20, -15-finger_gap_y, -10, 20, 20-finger_gap_y, 10])
            pullPointLocation=np.array([3, 8-finger_gap_y, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger1', finger_vol, finger_mesh, cable_file)
            )
            
            rotation=[-90, -65, 0]
            translation=np.array([-10.0, -5.0-finger_gap_y, 0.0])
            fixingBox = np.array([-20, -150-finger_gap_y, -10, 0, 20-finger_gap_y, 10])
            pullPointLocation=np.array([17, 0-finger_gap_y, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger2', finger_vol, finger_mesh, cable_file)
            )
            
            rotation=[90, -115, 0]
            translation = np.array([10.0, 10.0+finger_gap_y, 0.0])
            fixingBox = np.array([-20, -15+finger_gap_y, -10, 20, 20+finger_gap_y, 10])
            pullPointLocation=np.array([3, 8+finger_gap_y, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger3', finger_vol, finger_mesh, cable_file)
            )
            
            rotation=[-90, -65, 0]
            translation=np.array([-10.0, -5.0+finger_gap_y, 0.0])
            fixingBox = np.array([-20, -150+finger_gap_y, -10, 0, 20+finger_gap_y, 10])
            pullPointLocation=np.array([17, 0+finger_gap_y, 10.5])
            self._fingers.append(
                self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger4', finger_vol, finger_mesh, cable_file)
            )

        # vis_pts = np.array([
        #             [45, 0],
        #             [-45, 20],
        #             [-45, -20]
        #         ]) * self._gripper_size * 0.001 # array number measured in gripper_scale=1 and mm
    
    def load_finger(self, rotation, translation, fixingBox, pullPoint, name, finger_vol, finger_mesh, cable_file):
        translation = translation*self._gripper_size + self._position_offset + self._home_position
        fixingBox = fixingBox*self._gripper_size
        fixingBox[:3] = fixingBox[:3] + self._position_offset + self._home_position
        fixingBox[3:] = fixingBox[3:] + self._position_offset + self._home_position
        pullPointLocation=pullPoint*self._gripper_size + self._position_offset + self._home_position
        
        finger = self._root.addChild(name)
        eobject = finger.addChild(ElasticMaterialObject(finger,
                                    volumeMeshFileName=finger_vol,
                                    poissonRatio=self._poissonRatio,
                                    youngModulus=self._youngModulus,
                                    totalMass=0.5,
                                    surfaceColor=[0.0, 0.8, 0.7, 1],
                                    surfaceMeshFileName=finger_mesh,
                                    rotation=rotation,
                                    translation=translation.tolist(),
                                    collisionMesh=finger_mesh,
                                    scale=self._gripper_size,
                                    name='ElasticMO'))

        FixedBox(eobject, atPositions=fixingBox.tolist(), doVisualization=False)

        cable=PullingCable(eobject,
                    "PullingCable",
                    pullPointLocation=pullPointLocation.tolist(),
                    rotation=rotation,
                    translation=translation.tolist(),
                    uniformScale=self._gripper_size,
                    cableGeometry=loadPointListFromFile(cable_file))
        return finger

    
    def get_joint_limit(self):
        return np.stack([self._joint_limit_lower, self._joint_limit_upper], axis=0)
    
    def sample_joint(self, sample_range=1.):
        assert sample_range<=1.
        lower = self._joint_limit_lower
        upper = self._joint_limit_upper * sample_range
        joint_states = np.random.uniform(lower, upper)
        return joint_states
    
    # def grasp(self, pose, action):
    #     pos_x, pos_y, pos_z, angle = pose
    #     # # rotate
    #     # self.rotate(angle, np.pi/180)
    #     # # move on top of object
    #     # self.move((pos_x, pos_y, self._home_position[2]), v=1)
    #     # # move down
    #     # self.move((pos_x, pos_y, pos_z), v=1)
    #     self.step_pose(pose)
    #     # grasp
    #     self.step_joints(action)
    #     # up
    #     self.step_pose([pos_x, pos_y, self._home_position[2], angle])
        
    def reset(self, v=1.):
        self.move(self._home_position, v=v)
        self.rotate(0)
        self.step_joints(self._joint_limit_lower)
        
    def step_pose(self, pose, v=1.):
        pos_x, pos_y, pos_z, angle = pose
        pos_x, pos_y, pos_z = pos_x*self._unit_scale, pos_y*self._unit_scale, pos_z*self._unit_scale
        self.rotate(angle, np.pi/180)
        self.move((pos_x, pos_y, self._current_pos[2]), v=v)
        self.move((pos_x, pos_y, pos_z), v=v)
        for i in range(240):
            self.stepSimulation()
    
    
    def step_joints(self, target_states, v=.5):
        target_states = np.array(target_states)
        assert len(target_states)==self._joint_dof
        
        current_states = self.get_joint_states()
        if np.all(np.abs(target_states-current_states)<=1e-5):
            return
        n_step = int( np.abs(target_states-current_states).max()//v )
        trajectory = np.linspace(self._current_rot, target_states, n_step+1)
        for i in range(n_step):
            for j, finger in enumerate(self._fingers):
                finger.ElasticMO.PullingCable.CableConstraint.value = [trajectory[i+1, j]]
            self.stepSimulation()
        for i in range(240):
            self.stepSimulation()
    
    
    def up(self, z=None, v=1.):
        current_pos = self._current_pos
        if z is None:
            z = self._home_position[2]
        else:
            z *= self._unit_scale
        target_pos = [current_pos[0], current_pos[1], z]
        self.move(target_pos, v=v)
        
        
    def close(self, cable_step=1):
        is_closed = False
        while not is_closed:
            is_closed = True
            for i, finger in enumerate(self._fingers):
                v = finger.ElasticMO.PullingCable.CableConstraint.value.value.item()
                if v<=self._constraint_limit:
                    is_closed = False
                    finger.ElasticMO.PullingCable.CableConstraint.value = [v+cable_step]
            # print(v)
            self.stepSimulation()
            # print(v)
        for i in range(10):
            self.stepSimulation()
    
    def rotate(self, target_rot, v=np.pi/36):
        target_rot = target_rot%(2*np.pi)
        if self._current_rot==target_rot:
            return
        n_step = int( np.abs(target_rot-self._current_rot)//v )
        trajectory = np.linspace(self._current_rot, target_rot, n_step+1)
        
        finger_points = []
        pull_points = []
        for finger in self._fingers:
            finger_points.append(finger.ElasticMO.dofs.rest_position.value.copy())
            pull_points.append(finger.ElasticMO.PullingCable.CableConstraint.pullPoint.value.copy())
            
        for i in range(n_step):
            angle = trajectory[i+1] - trajectory[0]
            for j, finger in enumerate(self._fingers):
                finger.ElasticMO.dofs.rest_position = rotate_around_z(finger_points[j], angle).tolist()
                finger.ElasticMO.PullingCable.CableConstraint.pullPoint = rotate_around_z([pull_points[j]], angle)[0].tolist()
            self.stepSimulation()
        self._current_rot = target_rot
        
    def move(self, target_pos, v=1):
        target_pos = np.array(target_pos)
        distance = np.linalg.norm(target_pos - self._current_pos)
        if distance<=1e-5:
            return
        direction = (target_pos - self._current_pos)/distance
        n_step = int( distance//v )
        trajectory = np.linspace(0, distance, n_step+1)

        finger_points = []
        pull_points = []
        for finger in self._fingers:
            finger_points.append(finger.ElasticMO.dofs.rest_position.value.copy())
            pull_points.append(finger.ElasticMO.PullingCable.CableConstraint.pullPoint.value.copy())
        for i in range(n_step):
            distance_i = trajectory[i+1] - trajectory[0]
            for j, finger in enumerate(self._fingers):
                finger.ElasticMO.dofs.rest_position = translate_along_v(finger_points[j], direction, distance_i).tolist()
                finger.ElasticMO.PullingCable.CableConstraint.pullPoint = translate_along_v([pull_points[j]], direction, distance_i)[0].tolist()
            self.stepSimulation()
        self._current_pos = target_pos
    
    
    def get_joint_states(self):
        current_states = np.zeros(self._joint_dof)
        for i in range(self._joint_dof):
            current_states[i] = self._fingers[i].ElasticMO.PullingCable.CableConstraint.value.value.item()
        return current_states
    
    def set_root(self, root):
        root.addChild(self._root)
    
    # def get_name(self):
    #     return self._gripper_name
    
def rotate_around_z(points, angle):
    points = np.array(points)
    c = np.cos(angle)
    s = np.sin(angle)
    rot_mat = np.array([[c, s, 0],
                        [-s, c, 0],
                        [0, 0, 1]])
    rotated_points = points@rot_mat
    return rotated_points

def translate_along_v(points, v, d):
    points = np.array(points)
    translated_points = points + v*d
    return translated_points