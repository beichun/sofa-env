import os
import time

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

import Sofa.SofaGL

def init_display(display_size):
    pygame.display.init()
    pygame.display.set_mode(display_size, pygame.DOUBLEBUF | pygame.OPENGL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    Sofa.SofaGL.glewInit()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
def render(root):
    """
    Get the OpenGL Context to render an image (snapshot) of the simulation state
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cameraMVM = root.camera.getOpenGLModelViewMatrix()
    glMultMatrixd(cameraMVM)
    
    Sofa.SofaGL.draw(root)

    pygame.display.get_surface().fill((0,0,0))
    pygame.display.flip()
    
def get_tableau_palette(alpha=False):
    if alpha:
        palette = np.array([[ 78,121,167, 255], # blue
                            [255, 87, 89, 255], # red
                            [ 89,169, 79, 255], # green
                            [242,142, 43, 255], # orange
                            [237,201, 72, 255], # yellow
                            [176,122,161, 255], # purple
                            [255,157,167, 255], # pink 
                            [118,183,178, 255], # cyan
                            [156,117, 95, 255], # brown
                            [186,176,172, 255]  # gray
                            ],dtype=np.uint8)
    else:
        palette = np.array([[ 78,121,167], # blue
                            [255, 87, 89], # red
                            [ 89,169, 79], # green
                            [242,142, 43], # orange
                            [237,201, 72], # yellow
                            [176,122,161], # purple
                            [255,157,167], # pink 
                            [118,183,178], # cyan
                            [156,117, 95], # brown
                            [186,176,172]  # gray
                            ],dtype=np.uint8)
    return palette/255.0

def load_obj(bullet_client, scaling, position, orientation, visual_mesh, collision_mesh, color, fixed_base=False):
    template = """<?xml version="1.0" encoding="UTF-8"?>
<robot name="obj.urdf">
    <link name="baseLink">
        <contact>
            <lateral_friction value="1.0"/>
            <rolling_friction value="0.0001"/>
            <inertia_scaling value="3.0"/>
        </contact>
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value=".1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{0}" scale="1 1 1"/>
            </geometry>
            <material name="Cyan">
                <color rgba="{2} {3} {4} {5}"/>
            </material>
        </visual>
        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{1}" scale="1 1 1"/>
            </geometry>
        </collision>
    </link>
</robot>"""
    urdf_path = '.tmp_my_obj_%.8f%.8f.urdf' % (time.time(), np.random.rand())

    with open(urdf_path, "w") as f:
        f.write(template.format(visual_mesh, collision_mesh, color[0], color[1], color[2], color[3]))
    body_id = bullet_client.loadURDF(
        fileName=urdf_path,
        basePosition=position,
        baseOrientation=orientation,
        globalScaling=scaling,
        useFixedBase=fixed_base
    )
    os.remove(urdf_path)

    return body_id