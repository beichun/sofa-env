import os
import time
import math
from functools import partial
import shutil

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

import SofaRuntime
import Sofa
from stlib3.physics.rigid import Floor, Cube, Sphere, RigidObject

import sim.utils as utils
from sim.fusion import TSDFVolume, tsdf2mesh
from gripper_module import CableGripper, ArticulatedGripper

class SimAll(object):
    def __init__(self, gui_enabled=False):
        self._gui_enabled = gui_enabled
        
        self._sofa_backend = None
        
        # pybullet env
        self._bullet_client = bc.BulletClient(connection_mode=pybullet.GUI if self._gui_enabled else pybullet.DIRECT)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._bullet_client.setGravity(0, 0, -9.8)
        self.plane_id = self._bullet_client.loadURDF("plane.urdf")
        self._bullet_client.changeDynamics(self.plane_id, -1, lateralFriction=0.5)
        
        # data dir
        self._object_data_dir = os.path.abspath(os.path.join('assets', 'object'))
        
        # Defines where robot end effector can move to in world coordinates
        self._workspace_bounds = np.array([[-0.128, 0.128], # 3x2 rows: x,y,z cols: min,max
                                           [-0.128, 0.128],
                                           [ 0.000, 0.128]])
        wksp_center = self._workspace_bounds.mean(1)
        r = 0.1
        self._object_bound = np.array([[wksp_center[0]-r, wksp_center[0]+r],
                                        [wksp_center[1]-r, wksp_center[1]+r],
                                        [wksp_center[2]-r, wksp_center[2]+r]])
        
        # object and gripper
        self._sofa_object_list = []
        self._pybullet_object_list = []
        self._n_object = 0
        self._gripper = None
        self._gripper_home_position = np.array([-0., 0, 0.140])
        
        # observation
        self._voxel_size = 0.002
        self._heightmap_pix_size = self._voxel_size
        self._tsdf_bounds = np.array([[-0.2, 0.2], # 3x2 rows: x,y,z cols: min,max
                                      [-0.2, 0.2],
                                      [ 0.0, 0.2]])
        self._cam_lookat = self._tsdf_bounds.mean(axis=1)
        self._cam_image_size = (512, 512)
        self._cam_z_near = 0.01
        self._cam_z_far = 10.0
        self._cam_fov_w = 69.40
        self._cam_focal_length = (float(self._cam_image_size[1])/2)/np.tan((np.pi*self._cam_fov_w/180)/2)
        self._cam_fov_h = (math.atan((float(self._cam_image_size[0])/2)/self._cam_focal_length)*2/np.pi)*180
        self._cam_projection_matrix = self._bullet_client.computeProjectionMatrixFOV(
            fov=self._cam_fov_h,
            aspect=float(self._cam_image_size[1])/float(self._cam_image_size[0]),
            nearVal=self._cam_z_near,
            farVal=self._cam_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        self._cam_intrinsics = np.array([[self._cam_focal_length, 0, float(self._cam_image_size[1])/2],
                                                [0, self._cam_focal_length, float(self._cam_image_size[0])/2],
                                                [0, 0, 1]])
        fov_w = 69.4
        focal_length = (float(self._cam_image_size[1])/2)/np.tan((np.pi*fov_w/180)/2)
        fov_h = (math.atan((float(self._cam_image_size[0])/2)/focal_length)*2/np.pi)*180
        self._cam_znear = 0.01
        self._cam_zfar = 10.
        self._cam_projection_matrix = self._bullet_client.computeProjectionMatrixFOV(
            fov=fov_h,
            aspect=float(self._cam_image_size[1])/float(self._cam_image_size[0]),
            nearVal=self._cam_znear, farVal=self._cam_zfar
        )
        self._cam_intrinsics = np.array([[focal_length, 0, float(self._cam_image_size[1])/2],
                                            [0, focal_length, float(self._cam_image_size[0])/2],
                                            [0, 0, 1]])
        
        # sofa related
        self._unit_scale = 1000
        self._sofa_dt = 0.004
        SofaRuntime.importPlugin("SofaComponentAll")
        self._sofa_root = None
        # location used to export gripper meshes from sofa
        self._gripper_export_location = os.path.join('sim_logs', 'grippers', str(np.random.rand()))
        if os.path.exists(self._gripper_export_location):
            shutil.rmtree(self._gripper_export_location)
        os.makedirs(self._gripper_export_location)
        self._gripper_export_file = os.path.join(self._gripper_export_location, 'gripper')
        self._n_exported_meshes = 0
        
        self._grasp_offset = 0.03
        self._success_height = self._gripper_home_position[2] - 0.07
        
    def reset(self,
              n_object,
              object_category,
              object_id=None,
              object_size='random',
              gripper_type='cg',
              gripper_name=None,
              gripper_size=1.,
              **gripper_kwargs):
        # remove outdated obejcts and gripper
        for object_id in self._pybullet_object_list:
            self._bullet_client.removeBody(object_id)
        self._pybullet_object_list = []
        self._sofa_object_list = []
        if self._gripper is not None:
            self._bullet_client.removeBody(self._gripper_id)
            self._gripper = None
        
        # get objects
        generated_objects = self.generate_objects(n_object, object_category, object_size=object_size, size_lower=0.8, size_upper=1.)
        
        if isinstance(gripper_type, list):
            gripper_type = np.random.choice(gripper_type)
        sofa_gripper_types = ['cg']
        pybullet_gripper_types = ['articulated']
        if gripper_type in sofa_gripper_types:
            self._sofa_backend = True
        else:
            self._sofa_backend = False
            
        # load objects and gripper
        if self._sofa_backend:
            self.load_sofa(generated_objects, gripper_type, gripper_name, gripper_size, **gripper_kwargs)
        else:
            self.load_pybullet(generated_objects, gripper_type, gripper_name, gripper_size, **gripper_kwargs)
        
        # get object observation
        scene_observation = self.get_scene_observation()
        def get_z(x, y, tsdf_bounds, heightmap_pix_size, grasp_offset, depth_heightmap):
            # clip xy inside tsdf bound
            x = np.clip(x, tsdf_bounds[0, 0], tsdf_bounds[0, 1])
            y = np.clip(y, tsdf_bounds[1, 0], tsdf_bounds[1, 1])
            # pixel coordinate
            y_pix = int((y-tsdf_bounds[1,0]) / heightmap_pix_size)
            x_pix = int((x-tsdf_bounds[0,0]) / heightmap_pix_size)
            # get grasp height
            # utils.imwrite('depth.png', depth_heightmap)
            z = depth_heightmap[y_pix,x_pix] - grasp_offset
            return max(z, 0)
        observation = {
            'scene_init_pc': scene_observation['pc'],
            'object_bbox': scene_observation['bbox'],
            'get_z': partial(get_z,
                             tsdf_bounds=self._tsdf_bounds,
                             heightmap_pix_size=self._heightmap_pix_size,
                             grasp_offset=self._grasp_offset,
                             depth_heightmap=self.get_scene_heightmap()[1]),
            'gripper_type': gripper_type,
            'gripper_name': self._gripper.get_name(),
            'gripper_joint_limit': self._gripper.get_joint_limit()
        }
        return observation
        # np.savetxt(f'scene0.xyz', scene_observation['scene_pc'])
        # # get gripper observation
        # object_state = self.get_object_states()
        # self.set_object_states()
        # xy = scene_observation['object_bbox'].mean(0)
        # x, y = xy[0], xy[1]
        # z = self.get_grasp_height(x, y)
        # self._gripper.step_pose([x, y, z, 0])
        # gripper_observation = self.get_gripper_observation()
        # self._gripper.reset()
        # self.set_object_states(object_state)
        # scene_observation = self.get_scene_observation()
        # np.savetxt(f'scene1.xyz', scene_observation['scene_pc'])
        # return {**scene_observation, **gripper_observation}

    # def sample_action(self, sample_xy=True, sample_joint_range=1.):
    #     # sample pose
    #     pose = 0
        
    #     # sample joints
    #     joints = self._gripper.sample_joint(sample_joint_range)
    #     return pose, joints
        
    def step(self, pose, joints, record=False):
        observation = dict()
        
        if record:
            # 1. move objects away
            object_init_states = self.set_object_states()
            
            # 2. step gripper, get target
            self._gripper.step_pose(pose, v=0.5)
            self._gripper.step_joints(joints)
            obs = self.get_gripper_observation()
            observation['gripper_target_pc'] = obs['pc']
            
            # 3. move gripper to home position
            self._gripper.reset_home()
            self.set_object_states(object_init_states)
            
        # 4. step gripper pose, get gripper init
        self._gripper.step_pose(pose)
        if record:
            obs = self.get_gripper_observation()
            observation['gripper_init_pc'] = obs['pc']
            
        # 5. step gripper joints, get gripper final and scene final
        self._gripper.step_joints(joints)
        if record:
            obs = self.get_gripper_observation()
            observation['gripper_final_pc'] = obs['pc']
            obs = self.get_scene_observation()
            observation['scene_final_pc'] = obs['pc']
            
        # 6. move up gripper, get reward
        self._gripper.up()
        reward = (sum(self.get_reward())>=1)*1
        
        return reward, observation
        
    def get_reward(self):
        reward = []
        for i in range(self._n_object):
            if self._sofa_backend:
                h = self._sofa_object_list[i].mstate.position.value[0][2]/self._unit_scale
            else:
                h = self._bullet_client.getBasePositionAndOrientation(self._pybullet_object_list[i])[0][2]
            r = ( h>=self._success_height ) * 1
            reward.append(r)
        return reward
    
    
    def get_gt_action(self):
        actions = []
        for i in range(self._n_object):
            if self._sofa_backend:
                pos = self._sofa_object_list[i].mstate.position.value[0][:3]/self._unit_scale
            else:
                pos = self._bullet_client.getBasePositionAndOrientation(self._pybullet_object_list[i])[0]
            x, y = pos[0], pos[1]
            z = self.get_grasp_height(x, y)
            actions.append([x, y, z, 0])
        return actions
        # if self._sofa_backend:
        #     assert len(self._sofa_object_list)>0
        #     for object_node in self._sofa_object_list:
        #         x, y, z = self._object_list[0].mstate.position.value[0][:3]/self._unit_scale
        #         x = 0.04
        #         y = 0.07
        #         z = self.get_grasp_height(x, y)
        #         print('z', z)
        #         actions.append([x, y, z, 0])
        # else:
        #     assert len(self._pybullet_object_list)>0
        #     for obj_id in self._pybullet_object_list:
        #         pos = self._bullet_client.getBasePositionAndOrientation(obj_id)[0]
        #         x, y = pos[0], pos[1]
        # return actions
    
    
    def load_sofa(self, objects, gripper_type, gripper_name, gripper_size, **gripper_kwargs):
        self._sofa_root = Sofa.Core.Node("root")
        
        # create sofa graph
        self._sofa_root.addObject('VisualStyle', displayFlags="showVisualModels showBehaviorModels showInteractions")
        self._sofa_root.addObject("OglGrid", name="grid", plane="z", nbSubdiv=10, size=1000, draw=True)
        self._sofa_root.addObject("LightManager")
        self._sofa_root.addObject("DirectionalLight", direction=[0,1,1], shadowsEnabled=True)
        self._sofa_root.gravity = [0.0, 0.0, -9810]
        self._sofa_root.dt = self._sofa_dt
        # add plugins
        plugins=["SoftRobots", "SofaDeformable", "SofaEngine", 'SofaMiscCollision', 'SofaPython3']
        confignode = self._sofa_root.addChild("Config")
        for name in plugins:
            confignode.addObject('RequiredPlugin', name=name, printLog=False)
        # other components for simulation loop        
        # collision detection
        self._sofa_root.addObject('DefaultPipeline')
        self._sofa_root.addObject('BruteForceBroadPhase')
        self._sofa_root.addObject('BVHNarrowPhase')
        # contact
        self._sofa_root.addObject('RuleBasedContactManager', responseParams="mu="+str(0.5),
                                                        name='Response', response='FrictionContact')
        self._sofa_root.addObject('LocalMinDistance',
                        alarmDistance=4, contactDistance=3,
                        angleCone=0.01)
        # animation
        self._sofa_root.addObject('FreeMotionAnimationLoop')
        self._sofa_root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=1000)
        # camera
        self._sofa_root.addObject("InteractiveCamera", name="camera",
                             position=[0, -500, 500], lookAt=[0, 0, 0], distance=37,
                            fieldOfView=60, zNear=0.63, zFar=55.69)
        # add floor TODO: fix floor height
        self._floor = RigidObject(parent=self._sofa_root,
                                    name="Floor",
                                    uniformScale=1.,
                                    surfaceMeshFileName="assets/floor-trimesh1.obj",
                                    translation=[0.0,0.0,0],
                                    rotation=[0.0,0.0,0.0],
                                    inertiaMatrix=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
                                    color=utils.get_tableau_palette(True)[0].tolist(),
                                    isAStaticObject=True)
        
        # load objects in sofa
        object_parent = self._sofa_root.addChild('Objects')
        for i in range(len(objects)):
            object_info = objects[i]
            xyz = [i*self._unit_scale for i in object_info['xyz']]
            rot = self._bullet_client.getEulerFromQuaternion(object_info['quat'])
            object_node = RigidObject(object_info['name'],
                                    surfaceMeshFileName=object_info['collision_mesh'],
                                    translation=xyz*self._unit_scale,
                                    rotation=rot,
                                    uniformScale=object_info['scale']*self._unit_scale,
                                    totalMass=0.02,
                                    volume=20.,
                                    inertiaMatrix=[1000.0,0.0,0.0,0.0,1000.0,0.0,0.0,0.0,1000.0],
                                    color=object_info['color'],
                                    isAStaticObject=False,
                                    parent=object_parent)
            self._sofa_object_list.append(object_node)
        self._n_object = len(objects)
        # in pybullet
        for object_info in objects:
            object_id = self.load_object(object_info)
            self._pybullet_object_list.append(object_id)
        
        def stepSimulation(root, dt, gui_enabled, n_step=1):
            try:
                for i in range(n_step):
                    Sofa.Simulation.animate(root, dt)
                    Sofa.Simulation.updateVisual(root)
                    if gui_enabled:
                        utils.render(root)
            except KeyboardInterrupt:
                pass
        self.stepSim = partial(stepSimulation,
                                root=self._sofa_root,
                                dt=self._sofa_dt,
                                gui_enabled=self._gui_enabled)
        
        # load gripper
        gripper_root = self._sofa_root.addChild('Gripper')
        self._gripper = CableGripper(self._gripper_home_position*self._unit_scale, 
                                     gripper_size,
                                     gripper_name,
                                     gripper_root=gripper_root,
                                     stepSim=self.stepSim)
        self._mesh_exporter = gripper_root.addObject('OBJExporter', name='MeshExporter', filename=self._gripper_export_file, exportEveryNumberOfSteps=1, enable=False)
        self._n_exported_meshes = 0
        
        # initialize graph
        Sofa.Simulation.init(self._sofa_root)
        if self._gui_enabled:
            utils.init_display(display_size=(1920, 1080))
            Sofa.Simulation.initVisual(self._sofa_root)
            Sofa.Simulation.initTextures(self._sofa_root)
        
        for i in range(2*240):
            self.stepSim()
        
        self.sofa2pybullet()
        
    
    def load_pybullet(self, objects, gripper_type, gripper_name, gripper_size, **gripper_kwargs):
        raise NotImplementedError()
        
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
                object_dir = os.path.join(category_dir, selected_id)
                if object_category=='3dnet':
                    visual_mesh = os.path.join(object_dir, 'collision_vhacd.obj') # TODO: maybe use visual mesh?
                    collision_mesh = os.path.join(object_dir, 'collision_vhacd.obj')
                    initial_scale = 1.
                elif object_category=='primitive':
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
            x = np.random.rand() * np.diff(self._object_bound[0])[0] + self._object_bound[0, 0]
            y = np.random.rand() * np.diff(self._object_bound[1])[0] + self._object_bound[1, 0]
            x, y = 0.1, 0.1
            position = [x, y, 0.1*(i+1)]
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
    
    
    def sofa2pybullet(self):
        # gripper
        if self._n_exported_meshes>0:
            self._bullet_client.removeBody(self._gripper_id)
        self._mesh_exporter.enable = True
        self.stepSim()
        self._mesh_exporter.enable = False
        self._n_exported_meshes += 1
        gripper_mesh = f'{self._gripper_export_file}{self._n_exported_meshes:05d}.obj'
        self._gripper_id = utils.load_obj(self._bullet_client,
                             scaling=1/self._unit_scale,
                             position=[0, 0, 0],
                             orientation=self._bullet_client.getQuaternionFromEuler([0, 0, 0]),
                             visual_mesh=gripper_mesh,
                             collision_mesh=gripper_mesh,
                             color=[0.0, 0.8, 0.7, 1])
        os.remove(gripper_mesh)
        gripper_mtl = f'{self._gripper_export_file}{self._n_exported_meshes:05d}.mtl'
        os.remove(gripper_mtl)
        
        # objects
        assert len(self._sofa_object_list)==len(self._pybullet_object_list)
        for i in range(len(self._sofa_object_list)):
            pose = self._sofa_object_list[i].mstate.position.value[0]
            xyz = pose[:3] / self._unit_scale
            quat = pose[3:]
            self._bullet_client.resetBasePositionAndOrientation(self._pybullet_object_list[i], xyz, quat)
    
    
    def load_object(self, object_info):
        obj = utils.load_obj(self._bullet_client,
                             scaling=object_info['scale'],
                             position=object_info['xyz'],
                             orientation=object_info['quat'],
                             visual_mesh=object_info['visual_mesh'],
                             collision_mesh=object_info['collision_mesh'],
                             color=object_info['color'])
        return obj
    
    def get_cam_data(self, cam_position, cam_lookat, cam_up_direction):
        cam_view_matrix = self._bullet_client.computeViewMatrix(cam_position, cam_lookat, cam_up_direction)
        cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
        cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]

        camera_data = self._bullet_client.getCameraImage(
            self._cam_image_size[1],
            self._cam_image_size[0],
            cam_view_matrix,
            self._cam_projection_matrix,
            shadow=1,
            flags=self._bullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self._bullet_client.ER_BULLET_HARDWARE_OPENGL)
        rgb_pixels = np.array(camera_data[2]).reshape((self._cam_image_size[0], self._cam_image_size[1], 4))
        color_image = rgb_pixels[:,:,:3] # remove alpha channel
        z_buffer = np.array(camera_data[3]).reshape((self._cam_image_size[0], self._cam_image_size[1]))
        segmentation_mask = np.array(camera_data[4], np.int) # - not implemented yet with renderer=p.ER_BULLET_HARDWARE_OPENGL
        depth_image = (2.0*self._cam_znear*self._cam_zfar)/(self._cam_zfar+self._cam_znear-(2.0*z_buffer-1.0)*(self._cam_zfar-self._cam_znear))
        return color_image, depth_image, segmentation_mask, cam_pose_matrix
    
    def get_scene_heightmap(self):
        cam_position = [self._cam_lookat[0], self._cam_lookat[1], 1]
        cam_up_direction = [1, 0, 0]
        color_image, depth_image, segmentation_mask, cam_pose_matrix = self.get_cam_data(cam_position, self._cam_lookat, cam_up_direction)
        
        camera_points,color_points,segmentation_points = utils.get_pointcloud(color_image, depth_image, segmentation_mask, self._cam_intrinsics)
        camera_points = utils.transform_pointcloud(camera_points, cam_pose_matrix)
        color_heightmap,depth_heightmap,segmentation_heightmap = utils.get_heightmap(camera_points,
                                                                                     color_points,
                                                                                     segmentation_points,
                                                                                     self._tsdf_bounds,
                                                                                     self._heightmap_pix_size,
                                                                                     zero_level=self._tsdf_bounds[2,0])
        return color_heightmap, depth_heightmap, segmentation_heightmap
    
    def get_grasp_height(self, x, y, grasp_offset=0.03):
        # clip xy inside tsdf bound
        x = np.clip(x, self._tsdf_bounds[0, 0], self._tsdf_bounds[0, 1])
        y = np.clip(y, self._tsdf_bounds[1, 0], self._tsdf_bounds[1, 1])
        y_pix = int((y-self._tsdf_bounds[1,0]) / self._heightmap_pix_size)
        x_pix = int((x-self._tsdf_bounds[0,0]) / self._heightmap_pix_size)
        
        # get grasp height
        color_heightmap, depth_heightmap, segmentation_heightmap = self.get_scene_heightmap()
        # utils.imwrite('depth.png', depth_heightmap)
        z = depth_heightmap[y_pix,x_pix] - grasp_offset
        return max(z, 0)
    
    def get_tsdf(self):
        tsdf_obj = TSDFVolume(self._tsdf_bounds, voxel_size=self._voxel_size)
        # side
        home_position = self._cam_lookat
        cam_up_direction = [0, 0, 1]
        side_look_directions = np.linspace(0, 2*np.pi, num=8, endpoint=False)
        cam_distance = 1
        for direction in side_look_directions:
            cam_position = [
                home_position[0] + cam_distance * np.cos(direction),
                home_position[1] + cam_distance * np.sin(direction),
                home_position[2]
            ]
            color_image, depth_image, _, cam_pose_matrix = self.get_cam_data(cam_position, self._cam_lookat, cam_up_direction)
            tsdf_obj.integrate(color_image, depth_image, self._cam_intrinsics, cam_pose_matrix, obs_weight=1.)
        # top
        cam_position = [self._cam_lookat[0], self._cam_lookat[1], 1]
        up_direction = [1, 0, 0]
        color_image, depth_image, _, cam_pose_matrix = self.get_cam_data(cam_position, self._cam_lookat, up_direction)
        
        tsdf_obj.integrate(color_image, depth_image, self._cam_intrinsics, cam_pose_matrix, obs_weight=2.)
        tsdf_vol_cpu, _ = tsdf_obj.get_volume()
        tsdf_vol_cpu = np.transpose(tsdf_vol_cpu, [1, 0, 2]) # swap x-axis and y-axis to make it consitent with scene_tsdf
        # tsdf2mesh(tsdf_vol_cpu, 'scene.obj', skip_z=3)
        return tsdf_obj
    
    def get_scene_observation(self, n_point=8192):
        # make gripper transparent
        gripper_colors = [(x[1],x[7]) for x in self._bullet_client.getVisualShapeData(self._gripper_id)]
        for i in range(len(gripper_colors)):
            self._bullet_client.changeVisualShape(self._gripper_id, gripper_colors[i][0], rgbaColor=[0,0,0,0])
        scene_tsdf = self.get_tsdf()
        scene_pc, scene_mesh = scene_tsdf.get_point_cloud_sample(n_point, skip_z=3, upward_only=True, get_mesh=True)
        
        # Make gripper opaque again
        for i in range(len(gripper_colors)):
            self._bullet_client.changeVisualShape(self._gripper_id, gripper_colors[i][0], rgbaColor=gripper_colors[i][1])
        bbox_min = np.min(scene_pc[:, :2], axis=0)
        bbox_max = np.max(scene_pc[:, :2], axis=0)
        return {
            'pc': scene_pc,
            'bbox': np.stack([bbox_min, bbox_max], axis=0)
        }
    
    def get_gripper_observation(self, n_point=8192):
        # make object transparent
        object_colors = [self._bullet_client.getVisualShapeData(object_body_id)[0] for object_body_id in self._pybullet_object_list]
        for object_body_id in self._pybullet_object_list:
            self._bullet_client.changeVisualShape(object_body_id, -1, rgbaColor=[0,0,0,0])
        gripper_tsdf = self.get_tsdf()
        gripper_pc, scene_mesh = gripper_tsdf.get_point_cloud_sample(n_point, skip_z=3, upward_only=True, get_mesh=True)
        # Make object opaque again
        for i in range(len(object_colors)):
            self._bullet_client.changeVisualShape(self._pybullet_object_list[i], object_colors[i][1], rgbaColor=object_colors[i][7])
        return {
            'pc': gripper_pc
        }
    
    def get_object_states(self):
        object_states = []
        for i in range(self._n_object):
            if self._sofa_backend:
                state = self._sofa_object_list[i].mstate.position.value[0]
                pos = state[:3]/self._unit_scale
                quat = state[3:]
            else:
                state = self._bullet_client.getBasePositionAndOrientation(self._pybullet_object_list[i])
                pos = state[0]
                quat = state[1]
            object_states.append((pos, quat))
    
    def set_object_states(self, object_states=None):
        if object_states is None:
            object_states = [([1+i, 0, 0.1], [0., 0., 0., 1.]) for i in range(self._n_object)]
        for i in range(self._n_object):
            state = object_states[i]
            if self._sofa_backend:
                x, y, z = state[0]
                self._sofa_object_list[i].mstate.position = [[i*self._unit_scale for i in state[0]] + state[1]]
            else:
                self._bullet_client.resetBasePositionAndOrientation(self._pybullet_object_list[i], state[0], state[1])
        if self._sofa_backend:
            self.sofa2pybullet()
    
    def runSofa(self):
        # run sofa's own gui
        Sofa.Gui.GUIManager.Init("myscene", "qt")
        Sofa.Gui.GUIManager.createGUI(self._sofa_root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(self._sofa_root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI was closed")