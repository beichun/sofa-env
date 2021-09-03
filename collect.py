import time
import argparse
import multiprocessing as mp
import os
import shutil

import numpy as np
import h5py

from sim import Sim
from sim.fusion import tsdf2mesh, meshwrite


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default=None)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--gui', action='store_true')
parser.add_argument('--gripper_id', default=['articulated'], nargs='+')
parser.add_argument('--n_object', default=1, type=int)
parser.add_argument('--object_category',  default='primitive')
parser.add_argument('--n_env',  default=1, type=int)
parser.add_argument('--n_sample',  default=10, type=int)
parser.add_argument('--joint_sample_range', default=10., type=float)


def main():
    args = parser.parse_args()
    
    if args.gui:
        args.n_env = 1
    if os.path.exists(args.data_dir):
        if args.overwrite:
            shutil.rmtree(args.data_dir)
        else:
            raise OSError(f'Existing data in {args.data_dir}.')
    
    os.makedirs(args.data_dir)
    
    ctx = mp.get_context('spawn')
    task_q = ctx.Queue()
    [task_q.put(query) for query in range(args.n_sample*5)]
    finished_q = ctx.Queue(args.n_sample)
    processes = []
    for i in range(args.n_env):
        p = ctx.Process(target=sim_collect,
                        args=(args.gui,
                              args.n_object,
                              args.object_category,
                              args.gripper_id,
                              args.joint_sample_range/10.,
                              args.data_dir,
                              i,
                              task_q,
                              finished_q))
        p.daemon = True
        processes.append(p)
        
    for p in processes:
        p.start()
    
    n_finished = 0
    while True:
        if finished_q.qsize()>n_finished:
            n_finished = finished_q.qsize()
            print(f'Finished {finished_q.qsize()}/{args.n_sample}.')

        if finished_q.full():
            for p in processes:
                p.terminate()
            break
        
def sim_collect(gui, n_object, object_category, gripper_id, joint_sample_range, data_dir, env_idx, task_queue, finished_queue):
    
    env = Sim(gui)
    
    while not finished_queue.full():
        # collect data
        seed = task_queue.get()
        # print('env', env_idx, 'seed', seed)
        np.random.seed(seed)
        save_keys = ['gripper_id', 'scene_init_pc']
        try:
            obs1 = env.reset(n_object,
                            object_category,
                            gripper_id=gripper_id,
                            gripper_size=.6,
                            object_size='random')
            # env.runSofa()
        except RuntimeError as e:
            print(f'RuntimeError encountered in seed {seed} reset.')
            continue
        
        # sample pose action
        bbox = obs1['object_bbox']
        joint_limit = obs1['gripper_joint_limit']
        xy, rot, joint_states = sample(bbox, joint_limit, joint_sample_range)
        
        z = obs1['get_z'](xy[0], xy[1])
        pose = [xy[0], xy[1], z, rot]
        try:
            r, obs2 = env.step(pose, joint_states, True)
        except RuntimeError as e:
            print(f'RuntimeError encountered in seed {seed} step.')
            continue
        sim_result = {**obs1, **obs2}
        
        sample_dir = os.path.join(data_dir, str(seed))
        os.mkdir(sample_dir)
        dump_result(sim_result, sample_dir)
        
        finished_queue.put(seed)
        
        if finished_queue.full():
            break

def sample(bbox, joint_limit, joint_sample_range):
    xy = np.random.uniform(bbox[0], bbox[1])
    r = 2*np.pi*np.random.rand() - np.pi

    joint_states = np.random.uniform(joint_limit[0], joint_limit[1]*joint_sample_range)
    return xy, r, joint_states

def dump_result(result, save_dir):
    # save dataset
    save_keys = ['gripper_init_pc', 'gripper_final_pc', 'gripper_target_pc', 'scene_init_pc', 'scene_final_pc',
				'action_init', 'action_target', 'action_final']
    save_path = os.path.join(save_dir, 'original_data.hdf5')
    with h5py.File(save_path, 'w') as f:
        for key in save_keys:
            f.create_dataset(name=key, data=result[key], compression="gzip")
    
    # save .ply
    mesh_keys = ['gripper_init', 'gripper_final', 'gripper_target', 'scene_init', 'scene_final']
    for k in mesh_keys:
        mesh_path = os.path.join(save_dir, f'{k}_mesh.ply')
        meshwrite(mesh_path, *result[k+'_tsdf'].get_mesh(skip_z=3))
        
    # meshwrite('mesh_scene_init.ply', *obs1['scene_init_tsdf'].get_mesh(skip_z=3))
    

    
if __name__=='__main__':
    main()