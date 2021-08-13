import time

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

import Sofa
import Sofa.Gui
import SofaRuntime
from cablegripper.gripper import Gripper
from stlib3.physics.rigid import Floor, Cube, Sphere, RigidObject

from sim.sim_base import SimBase
import utils

class SofaSim(SimBase):
    def __init__(self, gui_enabled=False, n_object_available=5):
        super().__init__(gui_enabled)
        self._env_type = 'sofa'
        
        # sofa related
        self._unit_scale = 1000
        self._dt=0.004
        SofaRuntime.importPlugin("SofaComponentAll")
        self._root = Sofa.Core.Node("root")
        self._root.addObject('VisualStyle', displayFlags="showVisualModels")
        self._root.addObject("LightManager")
        self._root.addObject("DirectionalLight", direction=[0,1,1])
        self._root.gravity = [0., 0., -9.81*self._unit_scale]
        self._root.dt = self._dt
        
        # add plugins
        plugins=[
            "SoftRobots", "SofaDeformable", "SofaEngine",
            'SofaMiscCollision', 'SofaPython3']
        confignode = self._root.addChild("Config")
        for name in plugins:
            confignode.addObject('RequiredPlugin', name=name, printLog=False)
        # self._root.addObject('RequiredPlugin', pluginName="SofaImplicitOdeSolver SofaLoader SofaOpenglVisual SofaBoundaryCondition SofaGeneralLoader SofaGeneralSimpleFem") 
        
        # other components for simulation loop        
        # collision detection
        self._root.addObject('DefaultPipeline')
        # self._root.addObject('BruteForceDetection')
        self._root.addObject('BruteForceBroadPhase')
        self._root.addObject('BVHNarrowPhase')
        # contact
        self._root.addObject('DefaultContactManager', name="CollisionResponse", response="FrictionContact", responseParams="mu="+str(0.5))
        # self._root.addObject('RuleBasedContactManager', responseParams="mu="+str(0.5),
        #                                                 name='Response', response='FrictionContact')
        # self._root.addObject('LocalMinDistance',
        #                     alarmDistance=4, contactDistance=1,
        #                     angleCone=0.01)
        self._root.addObject('MinProximityIntersection',
                            alarmDistance=4, contactDistance=1)
        
        # animation
        self._root.addObject('FreeMotionAnimationLoop')
        self._root.addObject('DefaultVisualManagerLoop')
        # self._root.addObject('DefaultAnimationLoop')
        self._root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=500)
        
        # camera
        self._root.addObject("InteractiveCamera", name="camera", position=[0, 0, 400],
                            lookAt=[0,0,0], distance=37,
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
        self.n_object_available = n_object_available
        self._available_object_list = []
        for i in range(self.n_object_available):
            newSphere = Sphere(self._scene_node,
                               name='sphere'+str(i),
                        translation=[0.0,0.0,100*(i+1)],
                        rotation=[0.0,0.0,0.0],
                        inertiaMatrix=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
                        color=utils.get_tableau_palette(True)[0].tolist(),
                        isAStaticObject=False,
                        uniformScale=20)
            self._available_object_list.append(newSphere)

        Sofa.Simulation.init(self._root)
        # for object in self._available_object_list:
        #     object.mstate.reset()
        #     object.activated = True
        
        if self._gui_enabled:
            utils.init_display(display_size=(1920, 1080))
            Sofa.Simulation.initVisual(self._root)
            Sofa.Simulation.initTextures(self._root)
        
        # for observation
        self._bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        # Sofa.Simulation.print(self._root)
    
    def reset(self,
              n_object,
              object_category,
              object_id=None,
              object_size='random',
              size_lower=0.8,
              size_upper=1.,
              ):
        assert n_object<=len(self._available_object_list)
        # remove outdated obejcts and gripper
        # if self._scene_node is not None:
        #     # self._scene_node.cleanup() # https://github.com/sofa-framework/SofaPython3/blob/06824e802c236dc270f3fbafa017be4ec451bc91/docs/sphinx-stubs/Sofa/Core/__init__.pyi#L540
        #     self._root.removeChild(self._scene_node)
        for object in self._available_object_list:
            # object.mstate.reset()
            object.activated = False
        self._object_list = []
        self._gripper = None
        # dummy = self._scene_node.addObject('RequiredPlugin', name='SoftRobots', printLog=False)
        # object_root = dummy.getContext()
        
        # load objects
        generated_objects = self.generate_objects(n_object, object_category, object_id, object_size, size_lower, size_upper)

        for i in range(n_object):
            object_node = self._available_object_list[i]
            self.reset_object(object_node, generated_objects[i])
            object_node.activated = True
            self._object_list.append(object_node)
        #     # object_node.init()
        #     self._object_list.append(object_node)
        print(self._object_list)
            
        # # load gripper
        # Gripper(self._scene_node)
        
        # initialize graph again
        # self._scene_node.init()
        # Sofa.Simulation.reset(self._floor)
        # print(self._floor.visual.OglModel.material.getValueString())
        # Sofa.Simulation.print(self._root)
    
    def step(self):
        pass
    
    
    def get_reward(self):
        reward = []
        heights = []
        for object_node in self._object_list:
            h = object_node.mstate.position.value[0][2]
            v = object_node.mstate.velocity.value[0][2]
            heights.append(v)
        return (h, v)
    
    
    def get_gt_action(self):
        pass
    
    
    def reset_object(self, object_root, object_info):
        object_root.mstate.reset()
        xyz = [i*self._unit_scale for i in object_info['xyz']]
        # xyz = [0, 50, 150]
        orn = list(object_info['quat'])
        object_root.mstate.position = [xyz+orn]
        # print(object_root.mstate.velocity.value)
        print('before', object_root.visual.OglModel.position.value[0])
        object_root.mstate.showObjectScale = object_info['scale'] * self._unit_scale
        print(object_root.visual.loader.filename)
        object_root.visual.loader.filename = object_info['visual_mesh']
        object_root.visual.loader.scale = object_info['scale'] * self._unit_scale
        print(object_root.visual.loader.filename)
        object_root.init()
        # object_root.visual.OglModel.reset()
        print('after', object_root.visual.OglModel.position.value[0])
        # object_root.reset()
        print(object_root.visual.loader.filename)
        
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
        
    
    def stepSimulation(self, n_step=1):
        if self._gui_enabled:
            try:
                for i in range(n_step):
                    # print(i)
                    # if i==1:
                    #     print(self.get_reward())
                    Sofa.Simulation.animate(self._root, self._dt)
                    Sofa.Simulation.updateVisual(self._root)
                    utils.render(self._root)
                    time.sleep(self._dt)
            except KeyboardInterrupt:
                pass
        else:
            for i in range(n_step):
                for i in range(n_step):
                    # print(i)
                    # if i==0:
                    #     print(self.get_reward())
                    Sofa.Simulation.animate(self._root, self._dt)
                    Sofa.Simulation.updateVisual(self._root)
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