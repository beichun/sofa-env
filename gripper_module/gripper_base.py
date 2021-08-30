from abc import ABC, abstractmethod

class GripperBase(ABC):
    
    def __init__(self, home_position, gripper_size):
        self._home_position = home_position
        self._gripper_size = gripper_size
    
    @abstractmethod
    def sample_joint(self):
        pass
    
    @abstractmethod
    def step_pose(self):
        pass
    
    @abstractmethod
    def step_joints(self):
        pass
    
    @abstractmethod
    def get_joint_states(self):
        pass