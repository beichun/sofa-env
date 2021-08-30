import numpy as np

from gripper_module.gripper_base import GripperBase

class ArticulatedGripper(GripperBase):
    
    def __init__(self,
                 home_position,
                 gripper_size,
                 gripper_name,
                 bullet_client):
        super().__init__(home_position, gripper_size)
        self._gripper_category = 'cablegripper'
        self._home_position = home_position
        self._gripper_size = gripper_size
        
        self._position_offset = np.array([0, 0, 100.])*self._gripper_size
        
        self._fingers = []
        self.load_gripper()
    
    def load_gripper(self):
        rotation=[90, -115, 0]
        translation = np.array([10.0, 10.0, 0.0])
        fixingBox = np.array([-20, -15, -10, 20, 20, 10])
        pullPointLocation=np.array([3, 8, 10.5])
        self._fingers.append(
            self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger1')
        )

        rotation=[-90, -65, 0]
        translation=np.array([-10.0, -23.0, 0.0])
        fixingBox = np.array([-20, -30, -10, 20, 0, 0])
        pullPointLocation=np.array([3, -25, 10.5])
        self._fingers.append(
            self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger2')
        )

        rotation=[-90, -65, 0]
        translation=np.array([-10.0, 13.0, 0.0])
        fixingBox = np.array([-20, 5, -10, 20, 35, 10])
        pullPointLocation=np.array([3, 15, 10.5])
        self._fingers.append(
            self.load_finger(rotation, translation, fixingBox, pullPointLocation, 'Finger3')
        )
        
        vis_pts = np.array([
                    [45, 0],
                    [-45, 20],
                    [-45, -20]
                ]) * self._gripper_size * 0.001 # array number measured in gripper_scale=1 and mm
    
    def load_finger(self, rotation, translation, fixingBox, pullPoint, name):
        translation = translation*self._gripper_size + self._position_offset + self._home_position
        fixingBox = fixingBox*self._gripper_size
        fixingBox[:3] = fixingBox[:3] + self._position_offset + self._home_position
        fixingBox[3:] = fixingBox[3:] + self._position_offset + self._home_position
        pullPointLocation=pullPoint*self._gripper_size + self._position_offset + self._home_position
        f1 = Finger(self._root,
                    name,
                    rotation=rotation,
                    translation=translation.tolist(),
                    uniformScale=self._gripper_size,
                    fixingBox=fixingBox.tolist(),
                    pullPointLocation=pullPointLocation.tolist(),
                    poissonRatio=self._poissonRatio,
                    youngModulus=self._youngModulus
                    )
        
        
    def open(self):
        pass
        
    def close(self):
        pass
    
    def get_action_limit(self):
        pass
    
    def sample_action(self):
        pass
    
    def step(self):
        pass
    
    def get_state(self):
        pass