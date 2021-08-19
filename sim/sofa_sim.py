import time
from functools import partial

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

import Sofa
import Sofa.Gui
import SofaRuntime
# from cablegripper.gripper import Gripper
from gripper_module import CableGripper
from stlib3.physics.rigid import Floor, Cube, Sphere, RigidObject

from sim.sim_base import SimBase
import utils

class SofaSim(SimBase):
    def __init__(self, gui_enabled=False):
        super().__init__(gui_enabled)
        self._env_type = 'sofa'
        
        # sofa related
        self._unit_scale = 1000
        self._dt=0.01
        SofaRuntime.importPlugin("SofaComponentAll")
        self._root = Sofa.Core.Node("root")
        
        self.create_sofa_graph()
        
        # for observation
        self._bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        # Sofa.Simulation.print(self._root)
        @profile
        def stepSimulation(n_step, root, dt, gui_enabled):
            try:
                for i in range(n_step):
                    Sofa.Simulation.animate(root, dt)
                    Sofa.Simulation.updateVisual(root)
                    if gui_enabled:
                        utils.render(root)
                        # time.sleep(dt)
            except KeyboardInterrupt:
                pass
        self.stepSimulation = partial(stepSimulation,
                                      root=self._root,
                                      dt=self._dt,
                                      gui_enabled=self._gui_enabled)
    
    def reset(self,
              n_object,
              object_category,
              object_id=None,
              object_size='random',
              gripper_size=1.
              ):
        # remove outdated obejcts and gripper
        # for object_node in self._object_list:
        #     self._scene_node.cleanup() # https://github.com/sofa-framework/SofaPython3/blob/06824e802c236dc270f3fbafa017be4ec451bc91/docs/sphinx-stubs/Sofa/Core/__init__.pyi#L540
        #     self._scene_node.removeChild(object_node)
        self._root.removeChild(self._scene_node)
        self._object_list = []
        
        # load objects
        self._scene_node = self._root.addChild('MainScene')
        generated_objects = self.generate_objects(n_object, object_category, object_id, object_size, size_lower=0.8, size_upper=1.)

        for i in range(n_object):
            object_node = self.reset_object(self._scene_node, generated_objects[i])
            self._object_list.append(object_node)
            
        # load gripper
        self._gripper_node = self._scene_node.addChild('Gripper')
        def step_sim(step):
            self.stepSimulation(self._root, step)
        self._gripper = CableGripper(self._gripper_home_position*self._unit_scale, gripper_size, gripper_root=self._gripper_node, stepSim=self.stepSimulation)
        
        # initialize graph
        Sofa.Simulation.init(self._root)
        
        if self._gui_enabled:
            utils.init_display(display_size=(1920, 1080))
            Sofa.Simulation.initVisual(self._root)
            Sofa.Simulation.initTextures(self._root)
        
        
    def step(self, pose, action):
        x, y, z, angle = pose
        self._gripper.step_pose([x*self._unit_scale, y*self._unit_scale, z*self._unit_scale, angle])
        self._gripper.step_action(action)
        
    
    
    def get_reward(self):
        reward = []
        heights = []
        for object_node in self._object_list:
            h = object_node.mstate.position.value[0][2]
            print(object_node.mstate.position.value[0])
            r = ( h>=(self._success_height*self._unit_scale) ) * 1
            reward.append(r)
        return reward
    
    
    def get_gt_action(self):
        assert len(self._object_list)>0
        actions = []
        for object_node in self._object_list:
            x, y, z = self._object_list[0].mstate.position.value[0][:3]/self._unit_scale
            actions.append([x, y, z, np.pi/2])
        return actions
    
    def create_sofa_graph(self):
        self._root.addObject('VisualStyle', displayFlags="showVisualModels showBehaviorModels")
        self._root.createObject("OglGrid", name="grid", plane="z", nbSubdiv=10, size=1000, draw=True)
        self._root.addObject("LightManager")
        self._root.addObject("DirectionalLight", direction=[0,1,1], shadowsEnabled=True)
        self._root.gravity = [0., 0., -9.81*self._unit_scale]
        self._root.dt = self._dt
        
        # add plugins
        plugins=[
            "SoftRobots", "SofaDeformable", "SofaEngine",
            'SofaMiscCollision', 'SofaPython3', 'SofaOpenglVisual']
        confignode = self._root.addChild("Config")
        for name in plugins:
            confignode.addObject('RequiredPlugin', name=name, printLog=False)
        
        # other components for simulation loop        
        # collision detection
        self._root.addObject('DefaultPipeline')
        # self._root.addObject('BruteForceDetection')
        self._root.addObject('BruteForceBroadPhase')
        self._root.addObject('BVHNarrowPhase')
        # contact
        # self._root.addObject('DefaultContactManager', name="CollisionResponse", response="FrictionContact", responseParams="mu="+str(0.3))
        self._root.addObject('RuleBasedContactManager', responseParams="mu="+str(1.0),
                                                        name='Response', response='FrictionContact')
        # self._root.addObject('LocalMinDistance',
        #                     alarmDistance=5, contactDistance=1,
        #                     angleCone=0.01)
        self._root.addObject('MinProximityIntersection',
                            alarmDistance=5, contactDistance=1)
        
        # animation
        self._root.addObject('FreeMotionAnimationLoop')
        # self._root.addObject('DefaultVisualManagerLoop')
        self._root.addObject('GenericConstraintSolver', tolerance=1e-8, maxIterations=1000, computeConstraintForces=True)
        
        # camera
        self._root.addObject("InteractiveCamera", name="camera",
                             position=[-200, -200, 500], lookAt=[200,200,0], distance=37,
                            fieldOfView=60, zNear=0.63, zFar=55.69)
        self._floor = RigidObject(parent=self._root,
                        name="Floor",
                        surfaceMeshFileName="assets/floor/floor.obj",
                        translation=[0.0,0.0,-1.0],
                        rotation=[0.0,0.0,0.0],
                        inertiaMatrix=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
                        color=utils.get_tableau_palette(True)[0].tolist(),
                        isAStaticObject=True)
        # main scene node
        # all objects in the scene should register under this node
        self._scene_node = self._root.addChild('MainScene')
        
        
    def reset_object(self, object_parent, object_info):
        xyz = [i*self._unit_scale for i in object_info['xyz']]
        rot = self._bullet_client.getEulerFromQuaternion(object_info['quat'])
        object_node = RigidObject(object_info['name'],
                                  surfaceMeshFileName=object_info['collision_mesh'],
                                    translation=xyz,
                                    rotation=rot,
                                    uniformScale=object_info['scale']*self._unit_scale,
                                    totalMass=0.01,
                                    volume=1.,
                                    inertiaMatrix=[self._unit_scale, 0., 0., 0., self._unit_scale, 0., 0., 0., self._unit_scale],
                                    color=object_info['color'],
                                    isAStaticObject=False, parent=object_parent)
        return object_node
        
    def createNewSphere(self):
        obj = self._scene_node.addObject('RequiredPlugin', name='SoftRobots', printLog=False)
        root = obj.getContext()
        newSphere = root.addChild('FallingSphere')
        newSphere.addObject('EulerImplicitSolver')
        newSphere.addObject('CGLinearSolver', threshold='1e-09', tolerance='1e-09', iterations='200')
        MO = newSphere.addObject('MechanicalObject', showObject=True, position=[0, 0, 100, 0, 0, 0, 1], name=f'Particle', template='Rigid3d')
        Mass = newSphere.addObject('UniformMass', totalMass=1)
        Force = newSphere.addObject('ConstantForceField', name="CFF", totalForce=[0, -1, 0, 0, 0, 0] )
        Sphere = newSphere.addObject('SphereCollisionModel', name="SCM", radius=1.0 )
        
        newSphere.init()
        
    # @staticmethod
    # def stepSimulation(self, root, n_step=1):
    #     if self._gui_enabled:
    #         try:
    #             for i in range(n_step):
    #                 # print(i)
    #                 # if i==1:
    #                 #     print(self.get_reward())
    #                 Sofa.Simulation.animate(root, root.dt)
    #                 Sofa.Simulation.updateVisual(root)
    #                 utils.render(root)
    #                 time.sleep(root.dt)
    #         except KeyboardInterrupt:
    #             pass
    #     else:
    #         for i in range(n_step):
    #             for i in range(n_step):
    #                 # print(i)
    #                 # if i==0:
    #                 #     print(self.get_reward())
    #                 Sofa.Simulation.animate(root, root.dt)
    #                 Sofa.Simulation.updateVisual(root)
        
    def runSofa(self):
        # run sofa's own gui
        Sofa.Gui.GUIManager.Init("myscene", "qt")
        Sofa.Gui.GUIManager.createGUI(self._root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(self._root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI was closed")

def createNewSphere(root, z):
    # newSphere = root.addChild('FallingSphere')
    # newSphere.addObject('EulerImplicitSolver')
    # newSphere.addObject('CGLinearSolver', threshold='1e-09', tolerance='1e-09', iterations='200')
    # MO = newSphere.addObject('MechanicalObject', showObject=True, position=[0, 0, 100, 0, 0, 0, 1], name=f'Particle', template='Rigid3d')
    # Mass = newSphere.addObject('UniformMass', totalMass=1)
    # Force = newSphere.addObject('ConstantForceField', name="CFF", totalForce=[0, -1, 0, 0, 0, 0] )
    # Sphere = newSphere.addObject('SphereCollisionModel', name="SCM", radius=50.0 )
    # z = np.random.rand()*100+100
    newSphere = Sphere(root,
                        translation=[0.0,0.0,z],
                        rotation=[0.0,0.0,0.0],
                        inertiaMatrix=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
                        color=utils.get_tableau_palette(True)[0].tolist(),
                        isAStaticObject=False,
                        uniformScale=20)
    
    # newSphere.init()
    return newSphere