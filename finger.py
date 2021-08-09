# -*- coding: utf-8 -*-
import Sofa.Core
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from Sofa.Core import Node
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh

from splib3.loaders import loadPointListFromFile

class FingerController(Sofa.Core.Controller):
    def __init__(self, node, cable, *args, **kwargs):
        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.cableconstraintvalue = cable['CableConstraint.value']#.value
        self.name = "FingerController"

    def onKeypressedEvent(self, event):
        c = event['key']
        if (c == "+"):
            self.cableconstraintvalue.value =  self.cableconstraintvalue.value + 1.

        elif (c == "-"):
            displacement = self.cableconstraintvalue.value - 1.
            if(displacement < 0):
                displacement.fill(0)
            self.cableconstraintvalue.value = displacement

def Finger(parentNode=None, name="Finger",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
           fixingBox=[-5.0,0.0,0.0,10.0,15.0,20.0], pullPointLocation=[0.0,0.0,0.0]):

    finger = parentNode.addChild(name)
    eobject = finger.addChild(ElasticMaterialObject(finger,
                                   volumeMeshFileName="../data/mesh/finger.vtk",
                                   poissonRatio=0.3,
                                   youngModulus=18000,
                                   totalMass=0.5,
                                   surfaceColor=[0.0, 0.8, 0.7, 1],
                                   surfaceMeshFileName="../data/mesh/finger.stl",
                                   rotation=rotation,
                                   translation=translation,
                                   collisionMesh="../data/mesh/finger.stl"))

    FixedBox(eobject, atPositions=fixingBox, doVisualization=True)

    cable=PullingCable(eobject,
                 "PullingCable",
                 pullPointLocation=pullPointLocation,
                 rotation=rotation,
                 translation=translation,
                 cableGeometry=loadPointListFromFile("../data/mesh/cable.json"))

    finger.addObject(FingerController(eobject, cable))

    return finger

def createScene(rootNode):
    from stlib3.scene import MainHeader, ContactHeader
    MainHeader(rootNode, gravity=[0.0, -981.0, 0.0], plugins=["SoftRobots"])
    ContactHeader(rootNode, alarmDistance=4, contactDistance=3, frictionCoef=0.08)

    Finger(rootNode, translation=[1.0,0.0,0.0])
    return rootNode
