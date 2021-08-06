# -*- coding: utf-8 -*-

__all__=["RigidObject"]

from stlib3.physics.rigid.RigidObject import RigidObject

def Cube(node, **kwargs):
    """Create a rigid cube of unit dimension"""
    if "name" not in kwargs:
        kwargs["name"] = "Cube"
    return RigidObject(parent=node, surfaceMeshFileName="mesh/cube.obj", **kwargs)

def Sphere(node, **kwargs):
    """Create a rigid sphere of unit dimension"""
    if "name" not in kwargs:
        kwargs["name"] = "Sphere"
    return RigidObject(parent=node, surfaceMeshFileName="mesh/ball.obj", **kwargs)

def Floor(node, **kwargs):
    """Create a rigid floor of unit dimension"""
    if "name" not in kwargs:
        kwargs["name"] = "Floor"
    return RigidObject(parent=node, surfaceMeshFileName="mesh/floor.obj", **kwargs)

def createScene(rootNode):
    from stlib3.scene import MainHeader
    from stlib3.solver import DefaultSolver

    MainHeader(rootNode)
    DefaultSolver(rootNode)
    Cube(rootNode, translation=[5.0,0.0,0.0])
    Sphere(rootNode, translation=[-5.0,0.0,0.0])
    Floor(rootNode, translation=[0.0,-1.0,0.0])