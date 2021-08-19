from abc import ABC, abstractmethod
import os
import numpy as np

import utils

class SimBase(ABC):
    """SimBase base class for simulation environments.
    all environments should subclass this class.
    you should implement the following methods:
        - reset(): load objects in the scene
        - step(): step action, return reward
        - get_reward(): get reward
        - get_gt_action(): get ground truth action of an object

    """
    
    def __init__(self, gui_enabled):
        
        self._gui_enabled = gui_enabled
        
        self._object_data_dir = os.path.abspath(os.path.join('assets', 'object'))
        
        # Defines where robot end effector can move to in world coordinates
        self._workspace_bounds = np.array([[-0.128, 0.128], # 3x2 rows: x,y,z cols: min,max
                                           [-0.128, 0.128],
                                           [ 0.000, 0.128]])
        wksp_center = self._workspace_bounds.mean(1)
        r = 0.1
        self.position_bound = np.array([[wksp_center[0]-r, wksp_center[0]+r],
                                        [wksp_center[1]-r, wksp_center[1]+r],
                                        [wksp_center[2]-r, wksp_center[2]+r]])
        
        self._gripper_home_position = np.array([-0., 0, 0.3])
        self._success_height = self._gripper_home_position[-1] - 0.1
        # observation
        self._voxel_size = 0.002
        self._heightmap_pix_size = self._voxel_size
        
        self._object_list = []
        self._gripper = None
        
        
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def get_reward(self):
        pass
    
    @abstractmethod
    def get_gt_action(self):
        pass
    
    def get_cam_data(self):
        pass
        
    def get_observation(self, gui):
        print(gui)
    
    def generate_objects(self,
                         n_object,
                         object_category,
                         object_id=None,
                         object_size='random',
                         size_lower=0.8,
                         size_upper=1.):
        object_list = []
        category_dir = os.path.join(self._object_data_dir, object_category)
        
        for i in range(n_object):
            # id
            if object_id is None:
                if object_category=='3dnet':
                    available_ids = sorted(os.listdir(category_dir))
                elif object_category=='primitive':
                    available_ids = ['box', 'cross', 'cylinder', 'ell', 'hammer']
                else:
                    raise NotImplementedError(f'Not implemented object category: {object_category}')
                
                selected_id = np.random.choice(available_ids)
                # print(i, object_id)
                object_dir = os.path.join(category_dir, selected_id)
                if object_category=='3dnet':
                    visual_mesh = os.path.join(object_dir, 'collision_vhacd.obj') # TODO: maybe use visual mesh?
                    collision_mesh = os.path.join(object_dir, 'collision_vhacd.obj')
                    initial_scale = 1.
                else:
                    visual_mesh = os.path.join(object_dir, 'tinker.obj')
                    collision_mesh = visual_mesh
                    initial_scale = 0.0008
            else:
                selected_id = object_id
            # size
            if object_size=='random':
                random_size = size_lower + np.random.rand()*(size_upper-size_lower)
                generated_size = initial_scale*random_size
            else:
                generated_size = initial_scale*object_size
            # pose
            x = np.random.rand() * np.diff(self.position_bound[0])[0] + self.position_bound[0, 0]
            y = np.random.rand() * np.diff(self.position_bound[1])[0] + self.position_bound[1, 0]
            x, y = 0.1, 0.1
            position = [x, y, 0.08*(i+1)]
            orientation = self._bullet_client.getQuaternionFromEuler(2 * np.pi * np.random.rand(3))
            colors = utils.get_tableau_palette(True)
            color = colors[np.random.choice(len(colors))]
            object_list.append({
                'name': f'Object{i}',
                'object_category': object_category,
                'id': selected_id,
                'object_dir': object_dir,
                'visual_mesh': visual_mesh,
                'collision_mesh': collision_mesh,
                'scale': generated_size,
                'xyz': position,
                'quat': orientation,
                'color': color
            })
            
        return object_list