# -*- coding: utf-8 -*-
from stlib3.scene import MainHeader, ContactHeader
from stlib3.physics.rigid import Floor, Cube
from gripper import Gripper

def createScene(rootNode):
    """This is my first scene"""
    # MainHeader(rootNode, gravity=[0.0, -981.0, 0.0], plugins=["SoftRobots", "SofaDeformable", "SofaEngine"])
    rootNode.addObject('VisualStyle', displayFlags="showVisualModels")
    rootNode.gravity = [0.0, -981.0, 0.0]
    rootNode.dt=0.01

    plugins=[
        "SoftRobots", "SofaDeformable", "SofaEngine",
        'SofaMiscCollision', 'SofaPython3']
    confignode = rootNode.addChild("Config")
    for name in plugins:
        confignode.addObject('RequiredPlugin', name=name, printLog=False)
    
    # ContactHeader(rootNode, alarmDistance=4, contactDistance=3, frictionCoef=0.08)

    if rootNode.getChild("DefaultPipeline") is None:
            rootNode.addObject('DefaultPipeline')

    rootNode.addObject('BruteForceDetection')

    rootNode.addObject('RuleBasedContactManager', responseParams="mu="+str(0.08),
                                                    name='Response', response='FrictionContact')
    rootNode.addObject('LocalMinDistance',
                        alarmDistance=4, contactDistance=3,
                        angleCone=0.01)

    if rootNode.getChild("FreeMotionAnimationLoop") is None:
            rootNode.addObject('FreeMotionAnimationLoop')

    if rootNode.getChild("GenericConstraintSolver") is None:
            rootNode.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=1000)
            
    Gripper(rootNode)

    Floor(rootNode, name="Floor",
          color=[1.0,0.0,0.0, 1],
          translation=[0.0,-160.0,10.0],
          isAStaticObject=True)

    Cube(rootNode, name="Cube",
          uniformScale=20.0,
          color=[1.0,1.0,0.0, 1],
          totalMass=0.03,
          volume=20,
          inertiaMatrix=[1000.0,0.0,0.0,0.0,1000.0,0.0,0.0,0.0,1000.0],
          translation=[0.0,-130.0,10.0],
          isAStaticObject=False)
    
    # place light and a camera
    rootNode.addObject("LightManager")
    rootNode.addObject("DirectionalLight", direction=[0,1,0])
    rootNode.addObject("InteractiveCamera", name="camera", position=[0,15, 0],
                            lookAt=[0,0,0], distance=37,
                            fieldOfView=45, zNear=0.63, zFar=55.69)
    

    return rootNode
