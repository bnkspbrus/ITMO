import torch
import py360convert
import numpy as np
import os
from tqdm import tqdm
import os.path as osp
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from kitti360scripts.helpers.labels import trainId2label
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging as log

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

DATA_2D_EQUIRECT = "/home/jupyter/datasphere/project/ITMO/data_2d_equirect"
DATA_2D_SEMANTICS = "/home/jupyter/datasphere/project/ITMO/data_2d_semantics"

m2f_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
m2f_model = (Mask2FormerForUniversalSegmentation
             .from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
             .to(device))

of_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
of_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large").to(device)


class SidesDataset(Dataset):
    def __init__(self, sequence, image_lower_bound, image_upper_bound):
        self.sequence = sequence
        self.frames = os.listdir(osp.join(DATA_2D_EQUIRECT, sequence, 'image_02'))
        self.frames = list(filter(lambda x: x.endswith('.png'), self.frames))
        self.frames = list(
            filter(lambda x: image_lower_bound <= int(x.split('.')[0]) <= image_upper_bound, self.frames))
        # filter already processed frames
        self.frames = list(
            filter(lambda x: not osp.exists(osp.join(DATA_2D_SEMANTICS, sequence, 'semantic', x.split('.')[0], '0.png')),
                   self.frames))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image_02_path = osp.join(DATA_2D_EQUIRECT, self.sequence, 'image_02', frame)
        image_02 = Image.open(image_02_path)
        image_03_path = osp.join(DATA_2D_EQUIRECT, self.sequence, 'image_03', frame)
        if not osp.exists(image_03_path):
            return None
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

        return np.array(sides), int(frame.split('.')[0])


def process_result(i, j, result, sequence):
    instances = result["segmentation"]
    instances = instances.cpu().numpy()
    instances_path = osp.join(DATA_2D_SEMANTICS, sequence, 'instance', f'{i:010d}', f'{j}.png')
    os.makedirs(osp.dirname(instances_path), exist_ok=True)
    Image.fromarray(instances).save(instances_path)

    segments_info = result["segments_info"]
    id2labelId = np.zeros(len(segments_info) + 2, dtype=np.uint8)
    for segment_info in segments_info:
        id2labelId[segment_info["id"]] = segment_info["label_id"]
    label_ids = id2labelId[instances]
    semantics = trainId2labelId[label_ids]
    semantics_path = osp.join(DATA_2D_SEMANTICS, sequence, 'semantic', f'{i:010d}', f'{j}.png')
    os.makedirs(osp.dirname(semantics_path), exist_ok=True)
    Image.fromarray(semantics).save(semantics_path)

    semantics_rgb = trainId2color[label_ids]
    semantics_rgb_path = osp.join(DATA_2D_SEMANTICS, sequence, 'semantic_rgb', f'{i:010d}', f'{j}.png')
    os.makedirs(osp.dirname(semantics_rgb_path), exist_ok=True)
    Image.fromarray(semantics_rgb).save(semantics_rgb_path)


def process_sequence(sequence, image_lower_bound, image_upper_bound, model_name='mask2former'):
    log.debug(f"Processing panoptic segmentations: {sequence}/{image_lower_bound}_{image_upper_bound}")
    dataset = SidesDataset(sequence, image_lower_bound, image_upper_bound)
    loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    if model_name == 'mask2former':
        model = m2f_model
        processor = m2f_processor
        pre_process = lambda images: processor(images=images, return_tensors="pt")
    elif model_name == 'oneformer':
        model = of_model
        processor = of_processor
        pre_process = lambda images: processor(images=images, task_inputs=["panoptic"], return_tensors="pt")
    else:
        raise ValueError('Invalid model name')

    for sides, frame in tqdm(loader):
        if sides is None:
            continue
        images = list(sides.reshape(-1, 700, 700, 3))
        inputs = pre_process(images).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        results = processor.post_process_panoptic_segmentation(outputs, target_sizes=list(
            map(lambda x: x.shape[:2], images)))

        frame = frame.repeat_interleave(6)

        list(map(lambda x: process_result(x[0], x[1][0] % 6, x[1][1], sequence),
                 zip(frame, enumerate(results))))
        log.debug(f"Processing panoptic segmentations finished: {sequence}/{image_lower_bound}_{image_upper_bound}")


if __name__ == '__main__':
    process_sequence('2013_05_28_drive_0000_sync', 2, 385)
