import torch
import py360convert
import numpy as np
import os
import tqdm
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from kitti360scripts.helpers.labels import trainId2label
import matplotlib.pyplot as plt

trainId2labelId = np.zeros(256, dtype=np.uint8)
for trainId, label in trainId2label.items():
    trainId2labelId[trainId] = label.id

trainId2color = np.zeros((256, 3), dtype=np.uint8)
for trainId, label in trainId2label.items():
    trainId2color[trainId] = label.color

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

DATA_2D_EQUIRECT = "data_2d_equirect"
DATA_2D_SEMANTICS = "data_2d_semantics"

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
model = (Mask2FormerForUniversalSegmentation
         .from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
         .to(device))

for folder in os.listdir(DATA_2D_EQUIRECT):
    images = os.listdir(os.path.join(DATA_2D_EQUIRECT, folder, 'image_02'))
    min_frame = min(int(image.split('.')[0]) for image in images)
    max_frame = max(int(image.split('.')[0]) for image in images)
    for i in tqdm.tqdm(range(min_frame, max_frame + 1)):
        image_02 = Image.open(os.path.join(DATA_2D_EQUIRECT, folder, 'image_02', f'{i:010d}.png'))
        image_03_path = os.path.join(DATA_2D_EQUIRECT, folder, 'image_03', f'{i:010d}.png')
        if not os.path.exists(image_03_path):
            continue
        image_03 = Image.open(image_03_path)
        image_02 = np.roll(np.array(image_02), -350, axis=1)
        image_03 = np.roll(np.array(image_03), 1050, axis=1)
        image = np.copy(image_03)
        image[:, 350:1750] = image_02[:, 350:1750]
        cubes = [
            py360convert.e2c(image, face_w=700),
            py360convert.e2c(np.roll(image, -350, axis=1), face_w=700)
        ]
        sides = []
        for j in range(6):
            ii = j % 3
            cube = cubes[ii % 2]
            side = cube[700:1400, 1400 * (j // 3) + 700 * (ii // 2):1400 * (j // 3) + 700 * (ii // 2 + 1)]
            sides.append(side)
        inputs = processor(images=sides, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        results = processor.post_process_panoptic_segmentation(outputs, target_sizes=[side.shape[:2] for side in sides])

        for j, result in enumerate(results):
            instances = result["segmentation"]
            instances = instances.cpu().numpy()
            instances_path = os.path.join(DATA_2D_SEMANTICS, folder, 'instance', f'{i:010d}', f'{j}.png')
            os.makedirs(os.path.dirname(instances_path), exist_ok=True)
            Image.fromarray(instances).save(instances_path)

            segments_info = result["segments_info"]
            id2labelId = np.zeros(len(segments_info) + 2, dtype=np.uint8)
            for segment_info in segments_info:
                id2labelId[segment_info["id"]] = segment_info["label_id"]
            label_ids = id2labelId[instances]
            semantics = trainId2labelId[label_ids]
            semantics_path = os.path.join(DATA_2D_SEMANTICS, folder, 'semantic', f'{i:010d}', f'{j}.png')
            os.makedirs(os.path.dirname(semantics_path), exist_ok=True)
            Image.fromarray(semantics).save(semantics_path)

            semantics_rgb = trainId2color[label_ids]
            semantics_rgb_path = os.path.join(DATA_2D_SEMANTICS, folder, 'semantic_rgb', f'{i:010d}', f'{j}.png')
            os.makedirs(os.path.dirname(semantics_rgb_path), exist_ok=True)
            Image.fromarray(semantics_rgb).save(semantics_rgb_path)
