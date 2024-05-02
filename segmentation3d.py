from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d

DATA_2D_SEMANTICS = "data_2d_semantics"
DATA_3D_SEMANTICS = "data_3d_semantics"
KITTI_360 = 'KITTI_360'

S = 700
TRN = np.deg2rad(45)
F = 350
K = np.array([[F, 0, S / 2], [0, F, S / 2], [0, 0, 1]])
R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
R1 = np.array([[-np.cos(TRN), 0, -np.sin(TRN)], [0, 1, 0], [np.sin(TRN), 0, -np.cos(TRN)]])
BALL_RADIUS = 1000000


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


def process_ball(args):
    frameId, points, pcd_tree, folder, file, frame2pose, colors, cam_02, cam_03 = args
    curr_pose = np.matmul(cam_02.cam2world[frameId], np.linalg.inv(cam_02.camToPose))
    T = curr_pose[:3, 3]
    ball = pcd_tree.search_radius_vector_3d(T, BALL_RADIUS)
    vertices = points[ball[1]]
    vcolors = colors[ball[1]]

    points_local = world2cam(cam_03, vertices, frameId)
    u, v, depth = cam_00.cam2image(points_local)
    mask = (u >= 0) & (u < S) & (v >= 0) & (v < S)  # & (depth > 0)
    u, v, depth = u[mask], v[mask], depth[mask]
    # visualize
    image = np.zeros((S, S, 3), dtype=np.uint8)
    image[v, u] = vcolors[mask]
    plt.imshow(image)
    plt.show()


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
        process_ball((250, points, pcd_tree, folder, file, frame2pose, colors, cam_02, cam_03))
