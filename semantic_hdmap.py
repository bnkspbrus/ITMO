from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.labels import id2label
import os
import os.path as osp
import numpy as np
import open3d

DATA_3D_SEMANTICS = "/home/jupyter/datasphere/project/ITMO/data_3d_semantics"
KITTI_DATA_3D_SEMANTICS = "data_3d_semantics"
DATA_HDMAP = "/home/jupyter/datasphere/project/ITMO/data_hdmap"
KITTI_360 = '/home/jupyter/datasphere/project/KITTI-360'

labelId2color = np.zeros((256, 3), dtype=np.uint8)
for labelId, label in id2label.items():
    labelId2color[labelId] = label.color


def load_ball(folder, frame):
    ball0_path = osp.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}_0.npy')
    ball1_path = osp.join(DATA_3D_SEMANTICS, folder, 'ball', f'{frame:010d}_1.npy')
    if not osp.exists(ball0_path) or not osp.exists(ball1_path):
        return None
    ball0 = np.load(ball0_path)
    ball1 = np.load(ball1_path)
    return np.union1d(ball0, ball1)


def process_sequence(sequence, min_frame, max_frame):
    seq = int(sequence.split('_')[-2])
    file = '%010d_%010d.ply' % (min_frame, max_frame)
    semantics_path = osp.join(DATA_HDMAP, sequence, 'semantic',
                              '%04d_%010d_%010d.npy' % (seq, min_frame, max_frame))
    if osp.exists(semantics_path):
        return
    os.makedirs(osp.dirname(semantics_path), exist_ok=True)
    fpath = osp.join(KITTI_360, KITTI_DATA_3D_SEMANTICS, sequence, 'static', file)
    points = read_ply(fpath)
    points = np.array([points['x'], points['y'], points['z']]).T
    semantic3d = np.zeros((points.shape[0], 256), dtype=np.int32)
    for frame in range(min_frame, max_frame + 1):
        ball = load_ball(sequence, frame)
        if ball is None:
            continue
        spath = osp.join(DATA_3D_SEMANTICS, sequence, 'semantic', '%010d.npy' % frame)
        if not osp.exists(spath):
            continue
        semantics = np.load(spath)
        semantic3d[ball[semantics != 0], semantics[semantics != 0]] += 1
    semantic3d = np.argmax(semantic3d, axis=1).astype(np.uint8)
    os.makedirs(osp.dirname(semantics_path), exist_ok=True)
    np.save(semantics_path, semantic3d)


def draw_sequence(folder, file):
    ply_file = osp.join(KITTI_360, KITTI_DATA_3D_SEMANTICS, folder, 'static', file)
    points = read_ply(ply_file)
    points = np.array([points['x'], points['y'], points['z']]).T
    min_frame, max_frame = osp.splitext(file)[0].split('_')
    min_frame, max_frame = int(min_frame), int(max_frame)
    seq = int(folder.split('_')[-2])
    semantics_path = osp.join(DATA_HDMAP, folder, 'semantic',
                              '%04d_%010d_%010d.npy' % (seq, min_frame, max_frame))
    semantic3d = np.load(semantics_path)
    colors = labelId2color[semantic3d]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors / 255)
    open3d.visualization.draw_geometries_with_editing([pcd])


def main():
    process_sequence('2013_05_28_drive_0000_sync', 2, 385)


if __name__ == '__main__':
    # main()
    draw_sequence('2013_05_28_drive_0000_sync', '0000000002_0000000385.ply')
