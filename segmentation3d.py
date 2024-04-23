from kitti360scripts.helpers.ply import read_ply
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d
from PIL import Image

DATA_2D_SEMANTICS = "data_2d_semantics"
DATA_3D_SEMANTICS = "data_3d_semantics"
KITTI360_PATH = 'KITTI-360'

sequences = os.listdir(DATA_2D_SEMANTICS)

S = 700
FOV = np.deg2rad(90)
TRN = np.deg2rad(45)
F = S / (2 * np.tan(FOV / 2))
K = np.array([[F, 0, S / 2], [0, F, S / 2], [0, 0, 1]])
R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
R1 = np.array([[-np.cos(TRN), 0, -np.sin(TRN)], [0, 1, 0], [np.sin(TRN), 0, -np.cos(TRN)]])
BALL_RADIUS = 100


def world2cam(vertices, R, T):
    R_ = np.expand_dims(R, 0)
    T = np.reshape(T, [1, -1, 3])
    vertices = np.expand_dims(vertices, 0)
    points_local = np.matmul(R_.transpose(0, 2, 1), (vertices - T).transpose(0, 2, 1))
    return points_local[0]


def cam2image(points_local):
    points_local = np.expand_dims(points_local, 0)
    points_proj = np.matmul(K[:3, :3].reshape([1, 3, 3]), points_local)
    depth = points_proj[:, 2, :]
    depth[depth == 0] = -1e-6
    u = np.round(points_proj[:, 0, :] / np.abs(depth)).astype(np.int32)
    v = np.round(points_proj[:, 1, :] / np.abs(depth)).astype(np.int32)
    u, v, depth = u[0], v[0], depth[0]
    u, v = u.astype(np.int32), v.astype(np.int32)
    return u, v, depth


for seq in sequences:
    pose_dir = os.path.join(KITTI360_PATH, 'data_poses', seq)
    pose_file = os.path.join(pose_dir, "poses.txt")
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0]
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    cam2pose = {frame: pose for frame, pose in zip(frames, poses)}

    pcds = os.listdir(os.path.join(KITTI360_PATH, DATA_3D_SEMANTICS, 'train', seq, 'static'))
    pcds = sorted(pcds, key=lambda x: int(x.split('_')[0]))
    min_frame = int(pcds[0].split('_')[0])
    pcd_i = 0
    ply = read_ply(os.path.join(KITTI360_PATH, DATA_3D_SEMANTICS, 'train', seq, 'static', pcds[pcd_i]))
    points = np.array([ply['x'], ply['y'], ply['z']]).T
    colors = np.array([ply['red'], ply['green'], ply['blue']]).T
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd_tree = open3d.geometry.KDTreeFlann(pcd)

    for frame in frames:
        if frame < min_frame:
            continue
        if int(pcds[pcd_i].split('.')[0].split('_')[1]) < frame:
            if pcd_i + 1 >= len(pcds):
                break
            pcd_i += 1
            pcd = read_ply(os.path.join(KITTI360_PATH, DATA_3D_SEMANTICS, seq, 'static', pcds[pcd_i]))
            points = np.array([pcd['x'], pcd['y'], pcd['z']]).T
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd_tree = open3d.geometry.KDTreeFlann(pcd)
            colors = np.array([pcd['red'], pcd['green'], pcd['blue']]).T
        if not os.path.exists(os.path.join(DATA_2D_SEMANTICS, seq, 'semantic', '%010d' % frame)):
            continue
        if not os.path.exists(os.path.join(DATA_2D_SEMANTICS, seq, 'instance', '%010d' % frame)):
            continue
        curr_pose = cam2pose[frame]
        T = curr_pose[:3, 3]
        R = curr_pose[:3, :3]
        ball = pcd_tree.search_radius_vector_3d(T, BALL_RADIUS)
        vertices = points[ball[1]]
        points_local = world2cam(vertices, R, T)
        points_local = np.matmul(R0, points_local)
        for i in range(8):
            u, v, depth = cam2image(points_local)
            mask = (u >= 0) & (u < S) & (v >= 0) & (v < S) & (depth > 0)
            image = np.zeros((S, S, 3), dtype=np.uint8)
            u, v = u[mask], v[mask]
            image[v, u] = colors[ball[1]][mask]
            points_local = np.matmul(R1, points_local)
