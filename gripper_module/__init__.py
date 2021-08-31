import numpy as np

from gripper_module.CableGripper import CableGripper
from gripper_module.ArticulatedGripper import ArticulatedGripper

gripper_types = ['articulate', 'cg']

from gripper_module.articulated_gripper import gripper_names as articulated_names

cg_names = ['2f_short', '3f_short', '4f_short',
            '2f_long', '3f_long', '4f_long']


def parse_gripper_id(gripper_id):
    if isinstance(gripper_id, list):
        gripper_id = np.random.choice(gripper_id)
    assert isinstance(gripper_id, str)
    
    gripper_chars = gripper_id.split('-')
    gripper_type = gripper_chars[0]
    if len(gripper_chars)>1:
        gripper_name = gripper_chars[1]
    else:
        if gripper_type=='articulated':
            gripper_name = np.random.choice(articulated_names)
        elif gripper_type=='cg':
            gripper_name = np.random.choice(cg_names)
    return gripper_type, gripper_name