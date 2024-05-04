from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
from kitti360scripts.helpers.labels import id2label
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d

DATA_2D_SEMANTICS = "data_2d_semantics"
DATA_3D_SEMANTICS = "data_3d_semantics"
KITTI_360 = 'KITTI_360'

id2color = np.zeros((256, 3), dtype=np.uint8)
for id, label in id2label.items():
    id2color[id] = label.color

S = 700
F = 350
K = np.array([[F, 0, S / 2], [0, F, S / 2], [0, 0, 1]])
TRN = np.deg2rad(45)
R1 = np.array([[np.cos(TRN), 0, np.sin(TRN)], [0, 1, 0], [-np.sin(TRN), 0, np.cos(TRN)]]).T
BALL_RADIUS = 1000


class CameraPerspectiveV2(CameraPerspective):
    def __init__(self):
        self.K = K


cam_00 = CameraPerspectiveV2()


def get_frame2pose(folder):
    pose_file = os.path.join(KITTI_360, 'data_poses', folder, 'poses.txt')
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0]
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    return {frame: pose for frame, pose in zip(frames, poses)}


def get_kdtree(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    return open3d.geometry.KDTreeFlann(pcd)


def world2cam(cam, vertices, frameId):
    curr_pose = cam.cam2world[frameId]
    T = curr_pose[:3, 3]
    R = curr_pose[:3, :3]

    return cam.world2cam(vertices, R, T, True)


def process_halfball(cam, points, colors, frameId, side0, folder, semantic3d):
    points_local = world2cam(cam, points, frameId)
    points_local = R1.T @ points_local
    for side in range(side0, side0 + 3):
        u, v, depth = cam_00.cam2image(points_local)
        mask = (u >= 0) & (u < S) & (v >= 0) & (v < S) & (depth > 0)
        u, v, depth = u[mask], v[mask], depth[mask]
        semantic_path = os.path.join(DATA_2D_SEMANTICS, folder, 'semantic', f'{frameId:010d}', f'{side}.png')
        semantic = Image.open(semantic_path)
        semantic = np.array(semantic)
        semantic = semantic[v, u]
        semantic3d[mask, semantic] += 1
        points_local = R1 @ points_local


def process_ball(args):
    frameId, points, pcd_tree, folder, frame2pose, colors, cam_02, cam_03 = args
    if frameId not in frame2pose:
        return
    curr_pose = frame2pose[frameId]
    T = curr_pose[:3, 3]
    ball = pcd_tree.search_radius_vector_3d(T, BALL_RADIUS)
    ball = ball[1]
    points = points[ball]
    colors = colors[ball]

    os.makedirs(os.path.join(DATA_3D_SEMANTICS, folder, 'ball'), exist_ok=True)
    np.save(os.path.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frameId:010d}.npy'), ball)
    semantic3d = np.zeros((points.shape[0], 256), dtype=np.int32)
    process_halfball(cam_02, points, colors, frameId, 0, folder, semantic3d)
    process_halfball(cam_03, points, colors, frameId, 3, folder, semantic3d)
    semantic3d = np.argmax(semantic3d, axis=1).astype(np.uint8)
    os.makedirs(os.path.join(DATA_3D_SEMANTICS, folder, 'semantic'), exist_ok=True)
    np.save(os.path.join(DATA_3D_SEMANTICS, folder, 'semantic', f'{frameId:010d}.npy'), semantic3d)


def draw_semantic(frameId, folder):
    semantic3d = np.load(os.path.join(DATA_3D_SEMANTICS, folder, 'semantic', f'{frameId:010d}.npy'))
    ball = np.load(os.path.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frameId:010d}.npy'))
    statics = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static')
    ply = read_ply(os.path.join(statics, file))
    points = np.array([ply['x'], ply['y'], ply['z']]).T
    points = points[ball]
    colors = id2color[semantic3d]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors / 255)
    open3d.visualization.draw_geometries_with_editing([pcd])


for folder in os.listdir(DATA_2D_SEMANTICS):
    statics = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static')
    frame2pose = get_frame2pose(folder)
    cam_02 = CameraFisheye(KITTI_360, folder, 2)
    cam_03 = CameraFisheye(KITTI_360, folder, 3)

    for file in os.listdir(statics):
        ply = read_ply(os.path.join(statics, file))
        points = np.array([ply['x'], ply['y'], ply['z']]).T
        colors = np.array([ply['red'], ply['green'], ply['blue']]).T
        pcd_tree = get_kdtree(points)
        fname = os.path.splitext(file)[0]
        min_frame, max_frame = map(int, fname.split('_'))
        process_ball((250, points, pcd_tree, folder, frame2pose, colors, cam_02, cam_03))

draw_semantic(250, '2013_05_28_drive_0000_sync')
