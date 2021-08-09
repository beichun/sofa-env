import Sofa
import Sofa.SofaGL
import Sofa.Gui
import SofaRuntime
import Sofa.Simulation as sim
import os
import time
sofa_directory = os.environ['SOFA_ROOT']
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
from stlib3.physics.rigid import Floor, Cube
from stlib3.physics.rigid.RigidObject import RigidObject
from cablegripper import createScene as cgScene
# from liver_scene import createScene

display_size = (1920, 1080)


def init_display(node):
    pygame.display.init()
    pygame.display.set_mode(display_size, pygame.DOUBLEBUF | pygame.OPENGL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    Sofa.SofaGL.glewInit()
    Sofa.Simulation.initVisual(node)
    Sofa.Simulation.initTextures(node)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def simple_render(rootNode):
    """
     Get the OpenGL Context to render an image (snapshot) of the simulation state
     """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    ## maybe not needed?
    # glEnable(GL_LIGHTING)
    # glEnable(GL_DEPTH_TEST)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cameraMVM = rootNode.camera.getOpenGLModelViewMatrix()
    glMultMatrixd(cameraMVM)
    
    Sofa.SofaGL.draw(rootNode)

    s = pygame.display.get_surface().fill((0,0,0))
    pygame.display.flip()
    return glGetDoublev(GL_MODELVIEW_MATRIX)

if __name__ == '__main__':
    root = Sofa.Core.Node("myroot")
    SofaRuntime.importPlugin("SofaComponentAll")
    # createScene(root)
    cgScene(root)
    root.addObject("InteractiveCamera", name="camera", position=[0,15, 0],
                            lookAt=[0,0,0], distance=37,
                            fieldOfView=45, zNear=0.63, zFar=55.69)
    Sofa.Simulation.init(root)
    init_display(root)
    try:
        n=0
        while True:
            n+=1
            # print(n)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    print(pygame.key.name(event.key))
                # if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                #     print("Player moved up!")
            Sofa.Simulation.animate(root, root.getDt())
            Sofa.Simulation.updateVisual(root)
            mat_new = simple_render(root)
            if n >1:
                assert np.allclose(mat_new, mat_old, rtol=1e-07)
            mat_old = mat_new
            time.sleep(root.getDt())
            # time.sleep(1)
    except KeyboardInterrupt:
        pass
    # Sofa.Gui.GUIManager.Init("myscene", "qt")
    # Sofa.Gui.GUIManager.createGUI(root, __file__)
    # Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    # Sofa.Gui.GUIManager.MainLoop(root)
    # Sofa.Gui.GUIManager.closeGUI()
    # print("GUI was closed")