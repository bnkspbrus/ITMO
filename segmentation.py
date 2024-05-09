from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.labels import id2label
import os
import numpy as np
import open3d

DATA_3D_SEMANTICS = "data_3d_semantics"
DATA_SEMANTICS = "data_semantics"
KITTI_360 = 'KITTI_360'

labelId2color = np.zeros((256, 3), dtype=np.uint8)
for labelId, label in id2label.items():
    labelId2color[labelId] = label.color


def segment_sequence(folder, file):
    min_frame, max_frame = os.path.splitext(file)[0].split('_')
    min_frame, max_frame = int(min_frame), int(max_frame)
    seq = int(folder.split('_')[-2])
    semantics_path = os.path.join(DATA_SEMANTICS, '%04d_%010d_%010d.npy' % (seq, min_frame, max_frame))
    if os.path.exists(semantics_path):
        return
    ply_file = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static', file)
    points = read_ply(ply_file)
    points = np.array([points['x'], points['y'], points['z']]).T
    semantic3d = np.zeros((points.shape[0], 256), dtype=np.int32)
    for frame in range(min_frame, max_frame + 1):
        ball_path = os.path.join(DATA_3D_SEMANTICS, folder, 'ball', '%010d.npy' % frame)
        if not os.path.exists(ball_path):
            continue
        ball = np.load(ball_path)
        sem_path = os.path.join(DATA_3D_SEMANTICS, folder, 'semantic', '%010d.npy' % frame)
        if not os.path.exists(sem_path):
            continue
        semantics = np.load(sem_path)
        semantic3d[ball, semantics] += 1
    semantic3d = np.argmax(semantic3d, axis=1).astype(np.uint8)
    np.save(semantics_path, semantic3d)


def draw_sequence(folder, file):
    ply_file = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static', file)
    points = read_ply(ply_file)
    points = np.array([points['x'], points['y'], points['z']]).T
    min_frame, max_frame = os.path.splitext(file)[0].split('_')
    min_frame, max_frame = int(min_frame), int(max_frame)
    seq = int(folder.split('_')[-2])
    semantics_path = os.path.join(DATA_SEMANTICS, '%04d_%010d_%010d.npy' % (seq, min_frame, max_frame))
    semantic3d = np.load(semantics_path)
    colors = labelId2color[semantic3d]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors / 255)
    open3d.visualization.draw_geometries_with_editing([pcd])


def main():
    os.makedirs(DATA_SEMANTICS, exist_ok=True)
    for folder in os.listdir(DATA_3D_SEMANTICS):
        statics = os.path.join(KITTI_360, DATA_3D_SEMANTICS, 'train', folder, 'static')
        for file in os.listdir(statics):
            segment_sequence(folder, file)


if __name__ == '__main__':
    main()
    # draw_sequence('2013_05_28_drive_0000_sync', '0000000002_0000000385.ply')
