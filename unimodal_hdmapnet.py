from kitti360_config import WINDOWS, SEQUENCES
from fisheye2equirect import process_sequence as process_fisheye_sequence
from panoptic2d import process_sequence as process_panoptic_sequence
from semantic3d import process_sequence as process_3d_semantic_sequence
from semantic_hdmap import process_sequence as process_hdmap_semantic_sequence
from semantic2d import process_sequence as process_2d_semantic_sequence


def eval_semantic(model_name='segformer', merge_method='cluster'):
    for window in WINDOWS['val']:
        sequence = window.split('/')[0]
        min_frame, max_frame = map(int, window.split('/')[1].split('_'))
        rect_02, rect_03, frames = process_fisheye_sequence(sequence, min_frame, max_frame)
        if model_name == 'segformer':
            semantics, frames = process_2d_semantic_sequence(sequence, min_frame, max_frame, model_name=model_name,
                                                             rect_02=rect_02, rect_03=rect_03, frames=frames)
        elif model_name == 'mask2former' or model_name == 'oneformer':
            process_panoptic_sequence(sequence, min_frame, max_frame, model_name=model_name)
        else:
            raise ValueError('Invalid model name')
        process_3d_semantic_sequence(sequence, min_frame, max_frame)
        process_hdmap_semantic_sequence(sequence, min_frame, max_frame)


def main():
    eval_semantic()


if __name__ == '__main__':
    main()
