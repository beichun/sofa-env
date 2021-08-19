import time

import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

from sim.sim_base import SimBase
from gripper_module import ArticulatedGripper
import utils

class PybulletSim(SimBase):
    def __init__(self, gui_enabled=False):
        super().__init__(gui_enabled)
        self._env_type = 'pybullet'
        
        if gui_enabled:
            self._bullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._bullet_client.setGravity(0, 0, -9.8)

        self.plane_id = self._bullet_client.loadURDF("plane.urdf")
        self._bullet_client.changeDynamics(self.plane_id, -1, lateralFriction=0.5)
        
        self._bullet_client.resetDebugVisualizerCamera(0.6, 0, -92, [0, 0, 0])

        
        
    def reset(self,
              n_object,
              object_category,
              object_id=None,
              object_size='random',
              gripper_size=1.):
        # remove outdated obejcts and gripper
        for obj_id in self._object_list:
            self._bullet_client.removeBody(obj_id)
        self._object_list = []
        if self._gripper is not None:
            self._gripper.remove()
            self._gripper = None
        assert self._bullet_client.getNumBodies()==1 # should only have floor now
        
        # load objects
        generated_objects = self.generate_objects(n_object, object_category, object_id, object_size, size_lower, size_upper)

        for object_info in generated_objects:
            object_id = self.load_object(object_info)
            self._object_list.append(object_id)
        
        # load gripper
        ArticulatedGripper(self._gripper_home_position, gripper_size)
        
        
    
    def step(self):
        pass
    
    
    def get_reward(self):
        pass
    
    
    def get_gt_action(self):
        pass
    
    def load_object(self, object_info):
        obj = utils.load_obj(self._bullet_client,
                             scaling=object_info['scale'],
                             position=object_info['xyz'],
                             orientation=object_info['quat'],
                             visual_mesh=object_info['visual_mesh'],
                             collision_mesh=object_info['collision_mesh'],
                             color=object_info['color'])
        return obj
    
    def stepSimulation(self, n_step=1):
        for i in range(n_step):
            self._bullet_client.stepSimulation()
            time.sleep(0.01)
