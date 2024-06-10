import torch
import py360convert
import numpy as np
import os
from tqdm import tqdm
import os.path as osp
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
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

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768").to(device)


class SidesDataset(Dataset):
    def __init__(self, sequence, min_frame, max_frame, rect_02=None, rect_03=None, frames=None):
        self.sequence = sequence
        if frames is not None:
            self.frames = frames
        else:
            self.frames = os.listdir(osp.join(DATA_2D_EQUIRECT, sequence, 'image_02'))
            self.frames = list(filter(lambda x: x.endswith('.png'), self.frames))
            self.frames = list(map(lambda x: int(x.split('.')[0]), self.frames))
            self.frames = list(filter(lambda x: min_frame <= x <= max_frame, self.frames))
        self.rect_02 = rect_02
        self.rect_03 = rect_03

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if osp.exists(osp.join(DATA_2D_SEMANTICS, self.sequence, 'semantic', f'{frame:010d}')):
            return np.array([], ndmin=4), np.array([], ndmin=1), np.array([], ndmin=1)
        if self.rect_02 is not None and self.rect_03 is not None:
            image_02 = self.rect_02[idx]
            image_03 = self.rect_03[idx]
        else:
            image_02_path = osp.join(DATA_2D_EQUIRECT, self.sequence, 'image_02', '%010d.png' % frame)
            image_02 = Image.open(image_02_path)
            image_03_path = osp.join(DATA_2D_EQUIRECT, self.sequence, 'image_03', '%010d.png' % frame)
            if not osp.exists(image_03_path):
                return np.array([], ndmin=4), np.array([], ndmin=1), np.array([], ndmin=1)
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

        return np.array(sides), np.full(6, frame), np.arange(6)


def load_semantics(frame, sequence):
    semantics = []
    for i in range(6):
        path = osp.join(DATA_2D_SEMANTICS, sequence, 'semantic', f'{frame:010d}', f'{i}.png')
        semantics.append(np.array(Image.open(path)))
    return np.array(semantics)


def process_sequence(sequence, min_frame, max_frame, model_name='segformer', rect_02=None, rect_03=None, frames=None):
    log.debug(f"Processing panoptic segmentations: {sequence}/{min_frame}_{max_frame}")
    dataset = SidesDataset(sequence, min_frame, max_frame, rect_02, rect_03, frames)
    loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    data_frames = dataset.frames
    semantics = []
    frames = []
    indexes = []
    for sides, frames_, idxs in tqdm(loader):
        images = list(sides.reshape(-1, 700, 700, 3))
        inputs = feature_extractor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        result = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[(700, 700)] * len(images))
        semantics.extend(result)
        frames.extend(frames_)
        indexes.extend(idxs)

    for i, (frame, result, index) in enumerate(zip(frames, semantics, indexes)):
        os.makedirs(osp.join(DATA_2D_SEMANTICS, sequence, 'semantic', f'{frame:010d}'), exist_ok=True)
        os.makedirs(osp.join(DATA_2D_SEMANTICS, sequence, 'semantic_rgb', f'{frame:010d}'), exist_ok=True)
        Image.fromarray(result).save(osp.join(DATA_2D_SEMANTICS, sequence, 'semantic', f'{frame:010d}', f'{index}.png'))
        Image.fromarray(trainId2color[result]).save(
            osp.join(DATA_2D_SEMANTICS, sequence, 'semantic_rgb', f'{frame:010d}', f'{index}.png'))

    cached_frames = ~np.in1d(data_frames, frames)
    cached_semantics = np.apply_along_axis(load_semantics, 0, data_frames[cached_frames], sequence)

    frames = np.concatenate([frames, data_frames[cached_frames]])
    semantics = np.concatenate([semantics, cached_semantics])

    log.debug(f"Processing semantic segmentations finished: {sequence}/{min_frame}_{max_frame}")
    return semantics, frames


if __name__ == '__main__':
    process_sequence('2013_05_28_drive_0000_sync', 2, 385)
