from kitti360_config import WINDOWS, SEQUENCES
from fisheye2equirect import process_sequence as process_fisheye_sequence
from panoptic2d import process_sequence as process_panoptic_sequence
from semantic3d import process_sequence as process_3d_semantic_sequence
from semantic_hdmap import process_sequence as process_hdmap_semantic_sequence
from semantic2d import process_sequence as process_2d_semantic_sequence
from neighbours import search_neighbours
from kitti360scripts.helpers.ply import read_ply, write_ply
import os.path as osp
import numpy as np

KITTI_360 = "/home/jupyter/datasphere/project/KITTI-360/kitti360mm/raw"


def load_points(sequence, min_frame, max_frame):
    ply_path = osp.join(KITTI_360, 'data_3d_semantics', sequence, 'static', f'{min_frame:010d}_{max_frame:010d}.ply')
    ply = read_ply(ply_path)
    points = np.array([ply['x'], ply['y'], ply['z']]).T
    colors = np.array([ply['red'], ply['green'], ply['blue']]).T
    semantic = ply['semantic']
    instance = ply['instance']
    return points, colors, semantic, instance


def eval_semantic(model_name='segformer', merge_method='cluster'):
    for window in WINDOWS['val']:
        sequence = window.split('/')[0]
        min_frame, max_frame = map(int, window.split('/')[1].split('_'))
        points, colors, semantic, instance = load_points(sequence, min_frame, max_frame)
        rect_02, rect_03, frames = process_fisheye_sequence(sequence, min_frame, max_frame)
        if model_name == 'segformer':
            semantics, frames = process_2d_semantic_sequence(sequence, min_frame, max_frame, model_name=model_name,
                                                             rect_02=rect_02, rect_03=rect_03, frames=frames)
        elif model_name == 'mask2former' or model_name == 'oneformer':
            process_panoptic_sequence(sequence, min_frame, max_frame, model_name=model_name)
        else:
            raise ValueError('Invalid model name')
        frames = frames.int()
        neighbors_index = search_neighbours(points, sequence, frames, min_frame, max_frame)
        # process_3d_semantic_sequence(sequence, min_frame, max_frame)
        # process_hdmap_semantic_sequence(sequence, min_frame, max_frame)


def main():
    eval_semantic()


if __name__ == '__main__':
    main()
