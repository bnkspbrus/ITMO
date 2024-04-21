import os
import torch
import numpy as np
import cv2
import pycocotools.mask as maskUtils
from segformer import segformer_segmentation as segformer_func
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL
from kitti360scripts.helpers.labels import name2label


def inference(filename, output_path, img=None,
              semantic_branch_processor=None,
              semantic_branch_model=None,
              mask_branch_model=None,
              id2label=CONFIG_CITYSCAPES_ID2LABEL):
    anns = {'annotations': mask_branch_model.generate(img)}
    h, w, _ = img.shape
    class_names = []
    class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model)
    semantc_mask = class_ids.clone()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantc_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names

    sematic_class_in_img = torch.unique(semantc_mask)
    semantic_bitmasks, semantic_class_names = [], []

    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
        class_mask = semantc_mask == sematic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_class_names.append(class_name)
        semantic_bitmasks.append(class_mask)
        anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(
            np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = \
            anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')

    # mmcv.dump(anns, os.path.join(output_path, filename + '.json'))
    instances = np.zeros(img.shape[:2], dtype=np.uint16)
    semantics = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, annotation in enumerate(anns['annotations']):
        mask = maskUtils.decode(annotation['segmentation']).astype(bool)
        instances[mask] = i + 1
        semantics[mask] = name2label[annotation['class_name']].id
    os.makedirs(os.path.join(output_path, 'instance'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'semantic'), exist_ok=True)
    cv2.imwrite(os.path.join(output_path, 'instance', filename), instances)
    cv2.imwrite(os.path.join(output_path, 'semantic', filename), semantics)

    del img
    del anns
    del class_ids
    del semantc_mask
    del class_names
    del semantic_bitmasks
    del semantic_class_names
