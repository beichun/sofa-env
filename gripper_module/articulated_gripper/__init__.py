from gripper_module.articulated_gripper.barrett_hand import GripperBarrettHand

gripper_names = ['bh']
def fetch_gripper(gripper_name):
    if gripper_name=='bh' or gripper_name=='barrett_hand':
        return GripperBarrettHand