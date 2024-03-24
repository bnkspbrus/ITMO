import os
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from pipeline import inference

CKPT_PATH = 'ckpt/sam_vit_h_4b8939.pth'
OUT_DIR = 'data_2d_semantics'
DATA_DIR = 'data_2d_equirect'


def main():
    sam = sam_model_registry["vit_h"](checkpoint=CKPT_PATH).to(device=torch.device('cpu'))

    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        output_mode='coco_rle'
    )
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(device=torch.device('cpu'))

    for folder in os.listdir(DATA_DIR):
        for file in os.listdir(os.path.join(DATA_DIR, folder)):
            img = cv2.imread(os.path.join(DATA_DIR, folder, file))
            with torch.no_grad():
                inference(file,
                          OUT_DIR,
                          img=img,
                          semantic_branch_processor=semantic_branch_processor,
                          semantic_branch_model=semantic_branch_model,
                          mask_branch_model=mask_branch_model)


if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    main()
