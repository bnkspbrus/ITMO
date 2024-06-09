from kitti360_config import WINDOWS, SEQUENCES
from fisheye2equirect import process_sequence as process_fisheye_sequence
from panoptic2d import process_sequence as process_panoptic_sequence
from semantic3d import process_sequence as process_3d_semantic_sequence
from semantic_hdmap import process_sequence as process_hdmap_sequence


def eval_unimodal_hdmapnet(model_name='segformer', merge_method='cluster'):
    for window in WINDOWS['val']:
        sequence = window.split('/')[0]
        image_lower_bound, image_upper_bound = map(int, window.split('/')[1].split('_'))
        process_fisheye_sequence(sequence, image_lower_bound, image_upper_bound)
        if model_name == 'segformer':
            # throw not implemented error
            raise NotImplementedError
        elif model_name == 'mask2former' or model_name == 'oneformer':
            process_panoptic_sequence(sequence, image_lower_bound, image_upper_bound, model_name=model_name)
        else:
            raise ValueError('Invalid model name')
        process_3d_semantic_sequence(sequence, image_lower_bound, image_upper_bound)
        process_hdmap_sequence(sequence, image_lower_bound, image_upper_bound)


def main():
    eval_unimodal_hdmapnet()


if __name__ == '__main__':
    main()
