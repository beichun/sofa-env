def createScene(root):
    # Register all the common component in the factory.
    SofaRuntime.PluginRepository.addFirstPath(os.path.join(sofa_directory, 'bin'))
    root.addObject("RequiredPlugin", name="SofaOpenglVisual")  # visual stuff
    root.addObject("RequiredPlugin", name="SofaLoader")  # geometry loaders
    root.addObject("RequiredPlugin", name="SofaSimpleFem")  # diffusion fem
    root.addObject("RequiredPlugin", name="SofaBoundaryCondition")  # constraints
    root.addObject("RequiredPlugin", name="SofaEngine")  # Box Roi
    root.addObject("RequiredPlugin", name="SofaImplicitOdeSolver")  # implicit solver
    root.addObject("RequiredPlugin", name="SofaMiscForceField")  # meshmatrix
    root.addObject("RequiredPlugin", name="SofaGeneralEngine")  # TextureInterpolation
    root.addObject("RequiredPlugin", name="CImgPlugin")  # for loading a bmp image for texture
    root.addObject("RequiredPlugin", name="SofaBaseLinearSolver")
    root.addObject("RequiredPlugin", name="SofaGeneralVisual")
    root.addObject("RequiredPlugin", name="SofaTopologyMapping")
    root.addObject("RequiredPlugin", name="SofaGeneralTopology")
    root.addObject("RequiredPlugin", name="SofaGeneralLoader")

    ### these are just some things that stay still and move around
    # so you know the animation is actually happening
    root.gravity = [0, -1., 0]
    root.addObject("VisualStyle", displayFlags="showAll")
    root.addObject("MeshGmshLoader", name="meshLoaderCoarse",
                   filename="mesh/liver.msh")
    root.addObject("MeshObjLoader", name="meshLoaderFine",
                   filename="mesh/liver-smooth.obj")

    root.addObject("EulerImplicitSolver")
    root.addObject("CGLinearSolver", iterations="200",
                   tolerance="1e-09", threshold="1e-09")

    liver = root.addChild("liver")

    liver.addObject("TetrahedronSetTopologyContainer",
                    name="topo", src="@../meshLoaderCoarse")
    liver.addObject("TetrahedronSetGeometryAlgorithms",
                    template="Vec3d", name="GeomAlgo")
    liver.addObject("MechanicalObject",
                    template="Vec3d",
                    name="MechanicalModel", showObject="1", showObjectScale="3")

    liver.addObject("TetrahedronFEMForceField", name="fem", youngModulus="1000",
                    poissonRatio="0.4", method="large")

    liver.addObject("MeshMatrixMass", massDensity="1")
    liver.addObject("FixedConstraint", indices="2 3 50")
    visual = liver.addChild("visual")
    visual.addObject('MeshObjLoader', name="meshLoader_0", filename="mesh/liver-smooth.obj", handleSeams="1")
    visual.addObject('OglModel', name="VisualModel", src="@meshLoader_0", color='red')
    visual.addObject('BarycentricMapping', input="@..", output="@VisualModel", name="visual mapping")

    # place light and a camera
    root.addObject("LightManager")
    root.addObject("DirectionalLight", direction=[0,1,0])
    root.addObject("InteractiveCamera", name="camera", position=[0,15, 0],
                            lookAt=[0,0,0], distance=37,
                            fieldOfView=45, zNear=0.63, zFar=55.69)