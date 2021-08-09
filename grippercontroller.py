# -*- coding: utf-8 -*-
import Sofa.Core

def getTranslated(points, vec):
    r=[]
    for v in points:
        r.append( [v[0]+vec[0], v[1]+vec[1], v[2]+vec[2]] )
    return r

class GripperController(Sofa.Core.Controller):
    def __init__(self, node, fingers, *args, **kwargs):
        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.fingers = fingers
        self.name = "FingerController"

    def onKeypressedEvent(self, event):
        c = event['key']
        
        dir = None
        # UP key :
        if ord(c)==19:
            dir = [0.0,1.0,0.0]
        # DOWN key : rear
        elif ord(c)==21:
            dir = [0.0,-1.0,0.0]
        # LEFT key : left
        elif ord(c)==18:
            dir = [1.0,0.0,0.0]
        elif ord(c)==20:
            dir = [-1.0,0.0,0.0]

        if dir != None:
            for finger in self.fingers:
                # m = finger.getChild("ElasticMaterialObject")
                # mecaobject = finger['ElasticMaterialObject.dofs.rest_position']
                # mecaobject.findData('rest_position').value = getTranslated( mecaobject.rest_position,  dir )
                # mecaobject.rest_position.value = getTranslated( mecaobject.rest_position,  dir )

                # cable = m.getChild("PullingCable").getObject("CableConstraint")
                # p = cable.pullPoint[0]
                # cable.findData("pullPoint").value = [p[0]+dir[0], p[1]+dir[1], p[2]+dir[2]]
                
                eobject = finger['ElasticMaterialObject']
                mecaobject = eobject['dofs']
                mecaobject.rest_position.value = getTranslated( mecaobject.rest_position.value,  dir )

                cable = eobject['PullingCable.CableConstraint']
                p = cable.pullPoint
                cable["pullPoint"].value = [p[0]+dir[0], p[1]+dir[1], p[2]+dir[2]]
