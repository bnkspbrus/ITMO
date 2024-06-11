from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.labels import id2label
import os
import os.path as osp
import numpy as np
import open3d
from kitti360_config import CACHING_ENABLED, LOADING_ENABLED

DATA_3D_SEMANTICS = "data_3d_semantics"
KITTI_DATA_3D_SEMANTICS = "data_3d_semantics"
DATA_HDMAP = "data_hdmap"
KITTI_360 = "KITTI-360"
DATA_NEIGHBOURS = "data_neighbours"

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


def load_neighbors(sequence, min_frame, max_frame, frames):
    out_path = osp.join(DATA_NEIGHBOURS, sequence, f'{min_frame:010d}_{max_frame:010d}')
    neighbors_path = osp.join(out_path, 'neighbors.npy')
    frames_path = osp.join(out_path, 'frames.npy')
    frames0 = np.load(frames_path)
    assert np.all(frames0 == frames)
    return np.load(neighbors_path)


def process_sequence(sequence, min_frame, max_frame, points=None, semantics3d=None, neighbors_index=None):
    if points is None:
        file = f'{min_frame:010d}_{max_frame:010d}.ply'
        fpath = osp.join(KITTI_360, KITTI_DATA_3D_SEMANTICS, sequence, 'static', file)
        ply = read_ply(fpath)
        points = np.array([ply['x'], ply['y'], ply['z']]).T
    spath = osp.join(DATA_3D_SEMANTICS, sequence, 'semantic')
    frames = list(filter(lambda x: x.endswith('.npy'), os.listdir(spath)))
    frames = list(map(lambda x: int(osp.splitext(x)[0]), frames))
    frames = list(filter(lambda x: min_frame <= x <= max_frame, frames))
    if semantics3d is None:
        semantics3d = []
        for frame in frames:
            semantic = np.load(osp.join(spath, f'{frame:010d}.npy'))
            semantics3d.append(semantic)
        semantics3d = np.concatenate(semantics3d, axis=0)
    if neighbors_index is None:
        neighbors_index = load_neighbors(sequence, min_frame, max_frame, frames)
    semantic_hdmap = np.zeros((points.shape[0], 256), dtype=np.int32)
    for semantic3d, neighbors_idx in zip(semantics3d, neighbors_index):
        non_zero_mask = semantic3d != 0
        semantic_hdmap[neighbors_idx[non_zero_mask], semantic3d[non_zero_mask]] += 1
    semantic_hdmap = np.argmax(semantic_hdmap, axis=1)
    return semantic_hdmap


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
    main()
    # draw_sequence('2013_05_28_drive_0000_sync', '0000000002_0000000385.ply')
