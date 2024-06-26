from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
from kitti360scripts.helpers.labels import id2label
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import os.path as osp
import numpy as np
import open3d
import pandas as pd
import re
import pickle
from kitti360_config import LOADING_ENABLED, CACHING_ENABLED

DATA_2D_SEMANTICS = "/home/jupyter/datasphere/project/ITMO/data_2d_semantics"
DATA_3D_SEMANTICS = "/home/jupyter/datasphere/project/ITMO/data_3d_semantics"
DATA_NEIGHBOURS = "/home/jupyter/datasphere/project/ITMO/data_neighbours"
KITTI_DATA_3D_SEMANTICS = "data_3d_semantics"
DATA_3D_PROJECTION = "/home/jupyter/datasphere/project/ITMO/data_3d_projection"
KITTI_360 = "/home/jupyter/datasphere/project/KITTI-360/kitti360mm/raw"

id2color = np.zeros((256, 3), dtype=np.uint8)
for id, label in id2label.items():
    id2color[id] = label.color

S = 700
F = 350
K = np.array([[F, 0, S / 2], [0, F, S / 2], [0, 0, 1]])
TRN = np.deg2rad(45)
R1 = np.array([[np.cos(TRN), 0, np.sin(TRN)], [0, 1, 0], [-np.sin(TRN), 0, np.cos(TRN)]]).T


class CameraPerspectiveV2(CameraPerspective):
    def __init__(self):
        self.K = K


cam_00 = CameraPerspectiveV2()


def frames_poses(folder):
    pose_file = osp.join(KITTI_360, 'data_poses', folder, 'poses.txt')
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0].astype(np.int32)
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    return frames, poses


def get_kdtree(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    return open3d.geometry.KDTreeFlann(pcd)


def world2cam(cam, points, frame):
    curr_pose = cam.cam2world[frame]
    T = curr_pose[:3, 3]
    R = curr_pose[:3, :3]

    return cam.world2cam(points, R, T, True)


def z_buffer(u, v, depth):
    df = pd.DataFrame({'u': u, 'v': v, 'depth': depth})
    min_idx = df.groupby(['u', 'v'])['depth'].idxmin()
    mask = np.zeros_like(depth, dtype=bool)
    mask[min_idx] = True
    return u[mask], v[mask], mask


def project_vertices_halfball(cam, points, ball, frame, side0, folder):
    vertices = points[ball]
    points_local = world2cam(cam, vertices, frame)
    rotation = R1.T
    us = []
    vs = []
    masks = []
    for side in range(side0, side0 + 3):
        points_local = rotation @ points_local
        u, v, depth = cam_00.cam2image(points_local)
        mask = (u >= 0) & (u < S) & (v >= 0) & (v < S) & (depth > 0)
        u, v, depth = u[mask], v[mask], depth[mask]
        # u, v, dmask = z_buffer(u, v, depth)
        # mask[mask] = dmask
        os.makedirs(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}'), exist_ok=True)
        us.append(u)
        vs.append(v)
        masks.append(np.asarray(ball)[mask])
        rotation = R1 @ rotation
    return us, vs, masks


def project_vertices_ball(frame, points, folder, cam_02, cam_03, ball0, ball1):
    if osp.exists(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}')):
        with open(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}', 'us.pkl'), 'rb') as file:
            us = pickle.load(file)
        with open(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}', 'vs.pkl'), 'rb') as file:
            vs = pickle.load(file)
        with open(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}', 'masks.pkl'), 'rb') as file:
            masks = pickle.load(file)
        return us, vs, masks
    us0, vs0, masks0 = project_vertices_halfball(cam_02, points, ball0, frame, 0, folder)
    us1, vs1, masks1 = project_vertices_halfball(cam_03, points, ball1, frame, 3, folder)
    us = us0 + us1
    vs = vs0 + vs1
    masks = masks0 + masks1
    with open(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}', 'us.pkl'), 'wb') as file:
        pickle.dump(us, file)
    with open(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}', 'vs.pkl'), 'wb') as file:
        pickle.dump(vs, file)
    with open(osp.join(DATA_3D_PROJECTION, folder, f'{frame:010d}', 'masks.pkl'), 'wb') as file:
        pickle.dump(masks, file)
    return us, vs, masks


def segment_semantic_halfball(frame, side0, folder, semantic3d, us, vs, masks):
    for side in range(side0, side0 + 3):
        u = us[side]
        v = vs[side]
        mask = masks[side]
        semantic_path = osp.join(DATA_2D_SEMANTICS, folder, 'semantic', f'{frame:010d}', f'{side}.png')
        semantic = Image.open(semantic_path)
        semantic = np.array(semantic)
        semantic = semantic[v, u]
        mask = mask[semantic != 0]
        semantic3d[mask, semantic[semantic != 0]] += 1


def segment_semantic_ball(frame, points, colors, folder, ball0, ball1, us, vs, masks):
    semantic_path = osp.join(DATA_3D_SEMANTICS, folder, 'semantic', f'{frame:010d}.npy')
    if osp.exists(semantic_path):
        return
    semantic3d = np.zeros((points.shape[0], 256), dtype=np.int32)
    segment_semantic_halfball(frame, 0, folder, semantic3d, us, vs, masks)
    segment_semantic_halfball(frame, 3, folder, semantic3d, us, vs, masks)
    ball = np.union1d(ball0, ball1)
    semantic3d = semantic3d[ball]
    semantic3d = np.argmax(semantic3d, axis=1).astype(np.uint8)
    os.makedirs(osp.dirname(semantic_path), exist_ok=True)
    np.save(semantic_path, semantic3d)


def draw_semantic(frame, folder, file):
    cam_02 = CameraFisheye(KITTI_360, seq=folder, cam_id=2)
    cam_03 = CameraFisheye(KITTI_360, seq=folder, cam_id=3)
    semantic3d = np.load(osp.join(DATA_3D_SEMANTICS, folder, 'semantic', f'{frame:010d}.npy'))
    ball0 = np.load(osp.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}_0.npy'))
    ball1 = np.load(osp.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}_1.npy'))
    ball = np.union1d(ball0, ball1)
    statics = osp.join(KITTI_360, KITTI_DATA_3D_SEMANTICS, folder, 'static')
    ply = read_ply(osp.join(statics, file))
    points = np.array([ply['x'], ply['y'], ply['z']]).T
    colors = np.array([ply['red'], ply['green'], ply['blue']]).T
    points = points[ball]
    colors = colors[ball]
    colors = id2color[semantic3d]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors / 255)
    # point where the camera is
    curr_pose0 = cam_02.cam2world[frame]
    T0 = curr_pose0[:3, 3]
    # add point to the point cloud
    pcd.points.append(T0)
    pcd.colors.append([1, 0, 0])
    curr_pose1 = cam_03.cam2world[frame]
    T1 = curr_pose1[:3, 3]
    pcd.points.append(T1)
    pcd.colors.append([0, 1, 0])
    open3d.visualization.draw_geometries_with_editing([pcd])


def search_balls(curr_pose0, curr_pose1, points, folder, frame, radius):
    ball0_path = osp.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}_0.npy')
    ball1_path = osp.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}_1.npy')
    ball0, ball1 = None, None
    if osp.exists(ball0_path):
        ball0 = np.load(ball0_path)
    if osp.exists(ball1_path):
        ball1 = np.load(ball1_path)
    # curr_pose = np.reshape(curr_pose, [3, 4])
    T0 = curr_pose0[:3, 3]
    T1 = curr_pose1[:3, 3]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    # search ball with hidden_point_removal
    if ball0 is None:
        ball0 = pcd.hidden_point_removal(T0, radius)[1]
        os.makedirs(osp.dirname(ball0_path), exist_ok=True)
        np.save(ball0_path, ball0)
    if ball1 is None:
        ball1 = pcd.hidden_point_removal(T1, radius)[1]
        os.makedirs(osp.dirname(ball1_path), exist_ok=True)
        np.save(ball1_path, ball1)
    # ball = kdtree.search_radius_vector_3d(T, BALL_RADIUS)
    return ball0, ball1


def load_semantics(sequence, frame):
    semantics_path = osp.join(DATA_2D_SEMANTICS, sequence, 'semantic', f'{frame:010d}')
    semantics = []
    for i in range(6):
        path = osp.join(semantics_path, f'{i}.png')
        semantic = Image.open(path)
        semantic = np.array(semantic)
        semantics.append(semantic)
    return np.array(semantics)


def load_neighbors(sequence, min_frame, max_frame, frames):
    out_path = osp.join(DATA_NEIGHBOURS, sequence, f'{min_frame:010d}_{max_frame:010d}')
    neighbors_path = osp.join(out_path, 'neighbors.npy')
    frames_path = osp.join(out_path, 'frames.npy')
    frames0 = np.load(frames_path)
    assert np.all(frames0 == frames)
    return np.load(neighbors_path)


def process_cam_pose(cam_pose, sphere, semantics, side0, semantic3d, cam, frame):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(sphere)
    camera = cam_pose[:3, 3]
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pt_map = np.array(pt_map)
    sphere = sphere[pt_map]
    rotation = R1.T
    points_local = world2cam(cam, sphere, frame)
    for side in range(side0, side0 + 3):
        semantic = semantics[side]
        points_local = rotation @ points_local
        u, v, depth = cam_00.cam2image(points_local)
        mask = (u >= 0) & (u < S) & (v >= 0) & (v < S) & (depth > 0)
        non_zero_mask = semantic[u[mask], v[mask]] != 0
        mask[mask] = non_zero_mask
        u, v, depth = u[mask], v[mask], depth[mask]
        semantic3d[pt_map[mask], semantic[u, v]] += 1
        rotation = R1 @ rotation


def process_frame(sequence, frame, cam_pose0, cam_pose1, sphere, semantic, cam_02, cam_03):
    if LOADING_ENABLED and osp.exists(osp.join(DATA_3D_SEMANTICS, sequence, 'semantic', f'{frame:010d}.npy')):
        return np.load(osp.join(DATA_3D_SEMANTICS, sequence, 'semantic', f'{frame:010d}.npy'))
    semantic3d = np.zeros((sphere.shape[0], 256), dtype=np.int32)
    process_cam_pose(cam_pose0, sphere, semantic, 0, semantic3d, cam_02, frame)
    process_cam_pose(cam_pose1, sphere, semantic, 3, semantic3d, cam_03, frame)
    semantic3d = np.argmax(semantic3d, axis=1).astype(np.uint8)
    if CACHING_ENABLED:
        os.makedirs(osp.join(DATA_3D_SEMANTICS, sequence, 'semantic'), exist_ok=True)
        np.save(osp.join(DATA_3D_SEMANTICS, sequence, 'semantic', f'{frame:010d}.npy'), semantic3d)
    return semantic3d


class FramesDataset(Dataset):
    def __init__(self, sequence, cam_02, cam_03, min_frame, max_frame, points=None, frames=None, semantics=None,
                 neighbors_index=None):
        self.sequence, self.cam_02, self.cam_03 = sequence, cam_02, cam_03
        self.min_frame, self.max_frame = min_frame, max_frame
        cam_02 = CameraFisheye(KITTI_360, seq=sequence, cam_id=2)
        cam_03 = CameraFisheye(KITTI_360, seq=sequence, cam_id=3)
        self.cam2world0 = cam_02.cam2world
        self.cam2world1 = cam_03.cam2world
        if points is None:
            file = f'{min_frame:010d}_{max_frame:010d}.ply'
            path = osp.join(KITTI_360, KITTI_DATA_3D_SEMANTICS, sequence, 'static', file)
            ply = read_ply(path)
            self.points = np.array([ply['x'], ply['y'], ply['z']]).T
        else:
            self.points = points
        if frames is None:
            frame_pattern = re.compile(r'^\d{10}$')
            path = osp.join(DATA_2D_SEMANTICS, sequence, 'semantic')
            self.frames = list(filter(frame_pattern.match, os.listdir(path)))
            self.frames = list(map(int, self.frames))
            self.frames = list(filter(lambda x: min_frame <= x <= max_frame, self.frames))
            self.frames = list(filter(lambda x: x in self.cam2world0 and x in self.cam2world1, self.frames))
            self.semantics = None
            self.neighbors_index = load_neighbors(sequence, min_frame, max_frame, self.frames)
        else:
            self.frames = np.array(frames)
            mask = list(map(lambda x: x in self.cam2world0 and x in self.cam2world1, self.frames))
            self.frames = self.frames[mask]
            self.semantics = semantics[mask]
            self.neighbors_index = neighbors_index[mask]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        cam_pose0 = self.cam2world0[frame]
        cam_pose1 = self.cam2world1[frame]
        if self.semantics is not None:
            semantic = self.semantics[idx]
        else:
            semantic = load_semantics(self.sequence, frame)
        neighbors_idx = self.neighbors_index[idx]
        sphere = self.points[neighbors_idx]
        semantic = np.array(semantic, dtype=np.uint8)
        semantic3d = process_frame(self.sequence, frame, cam_pose0, cam_pose1, sphere, semantic, self.cam_02,
                                   self.cam_03)
        return semantic3d, frame


def process_sequence(sequence, min_frame, max_frame, points=None, frames=None, semantics=None, neighbors_index=None):
    cam_02 = CameraFisheye(KITTI_360, sequence, 2)
    cam_03 = CameraFisheye(KITTI_360, sequence, 3)
    dataset = FramesDataset(sequence, cam_02, cam_03, min_frame, max_frame, points, frames, semantics, neighbors_index)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    semantics3d = []
    frames0 = []
    for semantic3d, frame in tqdm(dataloader, total=len(dataloader)):
        semantics3d.append(semantic3d)
        frames0.append(frame)
    semantics3d = np.concatenate(semantics3d, axis=0)
    frames0 = np.concatenate(frames0, axis=0)
    return semantics3d, frames0


if __name__ == '__main__':
    process_sequence('2013_05_28_drive_0000_sync', 2, 385)
    # draw_semantic(251, '2013_05_28_drive_0000_sync', '0000000002_0000000385.ply')
