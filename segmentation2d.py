import torch
import py360convert
import numpy as np
import os
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from kitti360scripts.helpers.labels import trainId2label
import matplotlib.pyplot as plt

DATA_2D_EQUIRECT = "data_2d_equirect"
DATA_2D_SEMANTICS = "data_2d_semantics"

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")

pano = np.array(Image.open('data_2d_equirect/2013_05_28_drive_0000_sync/0000000000.png'))
cubes = [
    py360convert.e2c(pano, face_w=700),
    py360convert.e2c(np.roll(pano, -350, axis=1), face_w=700)
]
cubes = [np.roll(cube, -700, axis=1) for cube in cubes]
for i in range(8):
    cube = cubes[i % 2]
    image = cube[700:1400, 700 * (i // 2):700 * (i // 2 + 1)]
    image = Image.fromarray(image)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    instance_segmentation = result["segmentation"]
    instance_segmentation = np.array(instance_segmentation)
    instance_path = os.path.join(DATA_2D_SEMANTICS, '2013_05_28_drive_0000_sync', 'instance',
                                 '0000000000', str(i) + '.png')
    os.makedirs(os.path.dirname(instance_path), exist_ok=True)
    Image.fromarray(instance_segmentation).save(instance_path)

    semantic_segmentation = np.zeros((instance_segmentation.shape[0], instance_segmentation.shape[1]), dtype=np.int32)
    semantic_rgb = np.zeros((instance_segmentation.shape[0], instance_segmentation.shape[1], 3), dtype=np.uint8)
    for segment in result['segments_info']:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        label = trainId2label[segment_label_id]
        mask = instance_segmentation == segment_id
        semantic_segmentation[mask] = label.id
        semantic_rgb[mask] = label.color

    semantic_path = os.path.join(DATA_2D_SEMANTICS, '2013_05_28_drive_0000_sync', 'semantic',
                                 '0000000000', str(i) + '.png')
    os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
    Image.fromarray(semantic_segmentation).save(semantic_path)
    semantic_rgb_path = os.path.join(DATA_2D_SEMANTICS, '2013_05_28_drive_0000_sync', 'semantic_rgb',
                                     '0000000000', str(i) + '.png')
    os.makedirs(os.path.dirname(semantic_rgb_path), exist_ok=True)
    Image.fromarray(semantic_rgb).save(semantic_rgb_path)
