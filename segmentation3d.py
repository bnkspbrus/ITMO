from kitti360scripts.helpers.project import CameraFisheye
from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.labels import labels
import os
import numpy as np
import open3d

DATA_2D_SEMANTICS = "data_2d_semantics"
DATA_3D_SEMANTICS = "data_3d_semantics"
KITTI360_PATH = 'KITTI-360'

sequences = os.listdir(DATA_2D_SEMANTICS)

for sequence in sequences:
    cam = CameraFisheye(KITTI360_PATH, sequence)
    pcds = os.listdir(os.path.join(KITTI360_PATH, DATA_3D_SEMANTICS, 'train', sequence, 'static'))
    pcds = sorted(pcds, key=lambda x: int(x.split('_')[0]))
    min_frame = int(pcds[0].split('_')[0])
    pcd_i = 0
    pcd = read_ply(os.path.join(KITTI360_PATH, DATA_3D_SEMANTICS, 'train', sequence, 'static', pcds[pcd_i]))
    points = np.array([pcd['x'], pcd['y'], pcd['z']]).T
    colors = np.array([pcd['red'], pcd['green'], pcd['blue']]).T
    for frame in cam.frames:
        if frame < min_frame:
            continue
        if int(pcds[pcd_i].split('.')[0].split('_')[1]) < frame:
            if pcd_i + 1 < len(pcds):
                pcd_i += 1
                pcd = read_ply(os.path.join(DATA_3D_SEMANTICS, sequence, 'static', pcds[pcd_i]))
            else:
                break
