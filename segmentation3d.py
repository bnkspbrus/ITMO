from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
from kitti360scripts.helpers.labels import id2label
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import open3d
import pandas as pd
import re

DATA_2D_SEMANTICS = "data_2d_semantics"
DATA_3D_SEMANTICS = "data_3d_semantics"
DATA_3D_PROJECTION = "data_3d_projection"
KITTI_360 = 'KITTI_360'

id2color = np.zeros((256, 3), dtype=np.uint8)
for id, label in id2label.items():
    id2color[id] = label.color

S = 700
F = 350
K = np.array([[F, 0, S / 2], [0, F, S / 2], [0, 0, 1]])
TRN = np.deg2rad(45)
R1 = np.array([[np.cos(TRN), 0, np.sin(TRN)], [0, 1, 0], [-np.sin(TRN), 0, np.cos(TRN)]]).T
BALL_RADIUS = 100


class CameraPerspectiveV2(CameraPerspective):
    def __init__(self):
        self.K = K


cam_00 = CameraPerspectiveV2()


def frames_poses(folder):
    pose_file = os.path.join(KITTI_360, 'data_poses', folder, 'poses.txt')
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0].astype(np.int32)
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    return frames, poses


def get_kdtree(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    return open3d.geometry.KDTreeFlann(pcd)


def world2cam(cam, vertices, frameId):
    curr_pose = cam.cam2world[frameId]
    T = curr_pose[:3, 3]
    R = curr_pose[:3, :3]

    return cam.world2cam(vertices, R, T, True)


def z_buffer(u, v, depth):
    df = pd.DataFrame({'u': u, 'v': v, 'depth': depth})
    min_idx = df.groupby(['u', 'v'])['depth'].idxmin()
    mask = np.zeros_like(depth, dtype=bool)
    mask[min_idx] = True
    return u[mask], v[mask], mask


def project_vertices_halfball(cam, vertices, frameId, side0, folder):
    points_local = world2cam(cam, vertices, frameId)
    rotation = R1.T
    for side in range(side0, side0 + 3):
        points_local = rotation @ points_local
        u, v, depth = cam_00.cam2image(points_local)
        mask = (u >= 0) & (u < S) & (v >= 0) & (v < S) & (depth > 0)
        u, v, depth = u[mask], v[mask], depth[mask]
        u, v, dmask = z_buffer(u, v, depth)
        mask[mask] = dmask
        os.makedirs(os.path.join(DATA_3D_PROJECTION, folder, f'{frameId:010d}'), exist_ok=True)
        np.save(os.path.join(DATA_3D_PROJECTION, folder, f'{frameId:010d}', f'u_{side}.npy'), u)
        np.save(os.path.join(DATA_3D_PROJECTION, folder, f'{frameId:010d}', f'v_{side}.npy'), v)
        np.save(os.path.join(DATA_3D_PROJECTION, folder, f'{frameId:010d}', f'mask_{side}.npy'), mask)
        rotation = R1 @ rotation


def project_vertices_ball(frame, points, folder, cam_02, cam_03, ball):
    if os.path.exists(os.path.join(DATA_3D_PROJECTION, folder, f'{frame:010d}')):
        return
    points = points[ball]
    project_vertices_halfball(cam_02, points, frame, 0, folder)
    project_vertices_halfball(cam_03, points, frame, 3, folder)


def segment_semantic_halfball(frameId, side0, folder, semantic3d):
    for side in range(side0, side0 + 3):
        u = np.load(os.path.join(DATA_3D_PROJECTION, folder, f'{frameId:010d}', f'u_{side}.npy'))
        v = np.load(os.path.join(DATA_3D_PROJECTION, folder, f'{frameId:010d}', f'v_{side}.npy'))
        mask = np.load(os.path.join(DATA_3D_PROJECTION, folder, f'{frameId:010d}', f'mask_{side}.npy'))
        semantic_path = os.path.join(DATA_2D_SEMANTICS, folder, 'semantic', f'{frameId:010d}', f'{side}.png')
        semantic = Image.open(semantic_path)
        semantic = np.array(semantic)
        semantic = semantic[v, u]
        mask[mask] = semantic != 0
        semantic3d[mask, semantic[semantic != 0]] += 1


def segment_semantic_ball(frame, points, colors, folder, ball):
    semantic_path = os.path.join(DATA_3D_SEMANTICS, folder, 'semantic', f'{frame:010d}.npy')
    if os.path.exists(semantic_path):
        return
    points = points[ball]
    colors = colors[ball]
    semantic3d = np.zeros((points.shape[0], 256), dtype=np.int32)
    segment_semantic_halfball(frame, 0, folder, semantic3d)
    segment_semantic_halfball(frame, 3, folder, semantic3d)
    semantic3d = np.argmax(semantic3d, axis=1).astype(np.uint8)
    os.makedirs(os.path.join(DATA_3D_SEMANTICS, folder, 'semantic'), exist_ok=True)
    np.save(semantic_path, semantic3d)


def draw_semantic(frame, folder, file):
    semantic3d = np.load(os.path.join(DATA_3D_SEMANTICS, folder, 'semantic', f'{frame:010d}.npy'))
    ball = np.load(os.path.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}.npy'))
    statics = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static')
    ply = read_ply(os.path.join(statics, file))
    points = np.array([ply['x'], ply['y'], ply['z']]).T
    points = points[ball]
    colors = id2color[semantic3d]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors / 255)
    open3d.visualization.draw_geometries_with_editing([pcd])


def search_ball(pose, kdtree, folder):
    frame = int(pose[0])
    curr_pose = pose[1:]
    curr_pose = np.reshape(curr_pose, [3, 4])
    T = curr_pose[:3, 3]
    ball_path = os.path.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}.npy')
    if os.path.exists(ball_path):
        return
    ball = kdtree.search_radius_vector_3d(T, BALL_RADIUS)
    os.makedirs(os.path.dirname(ball_path), exist_ok=True)
    np.save(ball_path, ball[1])


class FrameDataset(Dataset):
    def __init__(self, folder, fname, frames, poses, cam_02, cam_03):
        self.folder, self.cam_02, self.cam_03 = folder, cam_02, cam_03
        min_frame, max_frame = map(int, os.path.splitext(fname)[0].split('_'))
        spath = os.path.join(DATA_2D_SEMANTICS, folder, 'semantic')
        patt = re.compile(r'^(\d{10})$')
        valid_frames = np.array(list(map(int, filter(patt.match, os.listdir(spath)))))
        valid_frames = valid_frames[(valid_frames >= min_frame) & (valid_frames <= max_frame)]
        valid_mask = np.isin(frames, valid_frames)
        self.frames = frames[valid_mask]
        poses = poses[valid_mask].reshape(-1, 3 * 4)
        fpath = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static', fname)
        ply = read_ply(fpath)
        self.points = np.array([ply['x'], ply['y'], ply['z']]).T
        self.colors = np.array([ply['red'], ply['green'], ply['blue']]).T
        kdtree = get_kdtree(self.points)
        np.apply_along_axis(search_ball, 1, np.column_stack((self.frames, poses)), kdtree, folder)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        ball = np.load(os.path.join(DATA_3D_SEMANTICS, self.folder, 'ball', f'{frame:010d}.npy'))
        project_vertices_ball(frame, self.points, self.folder, self.cam_02, self.cam_03, ball)
        segment_semantic_ball(frame, self.points, self.colors, self.folder, ball)
        return 0


def main():
    for folder in os.listdir(DATA_2D_SEMANTICS):
        statics = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static')
        frames, poses = frames_poses(folder)
        cam_02 = CameraFisheye(KITTI_360, folder, 2)
        cam_03 = CameraFisheye(KITTI_360, folder, 3)
        for file in os.listdir(statics):
            dataset = FrameDataset(folder, file, frames, poses, cam_02, cam_03)
            dataloader = DataLoader(dataset, batch_size=24, num_workers=4, shuffle=False)
            list(tqdm(dataloader, total=len(dataloader)))


if __name__ == '__main__':
    main()
    # draw_semantic(251, '2013_05_28_drive_0000_sync', '0000000002_0000000385.ply')
