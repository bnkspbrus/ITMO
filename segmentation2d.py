import torch
import py360convert
import numpy as np
import os
import tqdm
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from kitti360scripts.helpers.labels import trainId2label
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
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
        Image.fromarray(image_02).save('test_02.png')
        Image.fromarray(image_03).save('test_03.png')
        break
