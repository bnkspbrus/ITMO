import torch
import open3d.ml.torch as ml3d
import os.path as osp
import os
import numpy as np
from kitti360_config import CACHING_ENABLED

K = 10000
KITTI_360 = "/home/jupyter/datasphere/project/KITTI-360/kitti360mm/raw"
DATA_NEIGHBOURS = "/home/jupyter/datasphere/project/ITMO/data_neighbours"


def load_cam_poses(sequnce, frames0):
    pose_file = osp.join(KITTI_360, 'data_poses', sequnce, 'poses.txt')
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0].astype(np.int32)
    mapping = np.zeros(frames.max() + 1, dtype=np.int32)
    mapping[frames] = np.arange(frames.shape[0])
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    trans = poses[:, :3, 3]
    return trans[mapping[frames0]]


def search_neighbours(points, sequence, frames, min_frame, max_frame):
    if osp.exists(osp.join(DATA_NEIGHBOURS, sequence, f'{min_frame:010d}_{max_frame:010d}.npy')):
        return np.load(osp.join(DATA_NEIGHBOURS, sequence, f'{min_frame:010d}_{max_frame:010d}.npy'))
    points = torch.tensor(points)
    cam_poses = load_cam_poses(sequence, frames)
    cam_poses = torch.tensor(cam_poses)
    nsearch = ml3d.layers.KNNSearch()
    points = points.double()
    neighbors = nsearch(points, cam_poses, K)
    neighbors_index = neighbors.neighbors_index.reshape(-1, K)
    if CACHING_ENABLED:
        out_path = osp.join(DATA_NEIGHBOURS, sequence, f'{min_frame:010d}_{max_frame:010d}.npy')
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        np.save(out_path, neighbors_index)
    return neighbors_index
