from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.labels import labels
import os
import numpy as np
import open3d

KITTI_DATA_3D_SEMANTICS = "data_3d_semantics"
DATA_SEMANTICS = "data_semantics"
KITTI_360 = "KITTI_360"

valid_labels = []
for label in labels:
    if label.trainId < 0 or label.id == 0:
        continue
    # we ignore the following classes during evaluation
    if label.name in ['train', 'bus', 'rider', 'sky']:
        continue
    # we append all found labels, regardless of being ignored
    valid_labels.append(label.id)
valid_labels = list(set(valid_labels))


def cluster_sequence(points, colors, semantic3d, semantic_path, instance_path):
    instance3d = np.zeros(points.shape[0], dtype=np.int32)
    for label in valid_labels:
        mask = (semantic3d == label) | (semantic3d == 0)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[mask])
        pcd.colors = open3d.utility.Vector3dVector(colors[mask] / 255)
        with open3d.utility.VerbosityContextManager(
                open3d.utility.VerbosityLevel.Debug) as cm:
            labels_ = np.array(
                pcd.cluster_dbscan(eps=0.2, min_points=100, print_progress=True))
        labels_ += instance3d.max() + 1
        for label_ in np.unique(labels_):
            if label_ == 0:
                continue
            mask_ = np.zeros(mask.shape[0], dtype=bool)
            mask_[mask] = labels_ == label_
            if (semantic3d[mask_] == label).sum() < 100:
                continue
            instance3d[mask_] = label_
            semantic3d[mask_] = label
    os.makedirs(os.path.dirname(instance_path), exist_ok=True)
    np.save(instance_path, instance3d)
    np.save(semantic_path, semantic3d)


def load_points_semantic(folder, file):
    ply_file = os.path.join(KITTI_360, KITTI_DATA_3D_SEMANTICS, 'train', folder, 'static', file)
    ply = read_ply(ply_file)
    points = np.array([ply['x'], ply['y'], ply['z']]).T
    colors = np.array([ply['red'], ply['green'], ply['blue']]).T
    min_frame, max_frame = os.path.splitext(file)[0].split('_')
    min_frame, max_frame = int(min_frame), int(max_frame)
    seq = int(folder.split('_')[-2])
    semantics_path = os.path.join(DATA_SEMANTICS, folder, 'semantic', '%04d_%010d_%010d.npy' % (seq, min_frame, max_frame))
    semantic3d = np.load(semantics_path)
    instances_path = os.path.join(DATA_SEMANTICS, folder, 'instance', '%04d_%010d_%010d.npy' % (seq, min_frame, max_frame))
    return points, colors, semantic3d, semantics_path, instances_path


def main():
    os.makedirs(DATA_SEMANTICS, exist_ok=True)
    for folder in os.listdir(DATA_SEMANTICS):
        statics = os.path.join(KITTI_360, KITTI_DATA_3D_SEMANTICS, 'train', folder, 'static')
        for file in os.listdir(statics):
            points, colors, semantic3d, sem_path, ins_path = load_points_semantic(folder, file)
            cluster_sequence(points, colors, semantic3d, sem_path, ins_path)


if __name__ == '__main__':
    main()
