import torch
import py360convert
import numpy as np
import os
import tqdm
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from kitti360scripts.helpers.labels import trainId2label
from torch.utils.data import Dataset, DataLoader
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


class SidesDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = os.listdir(os.path.join(DATA_2D_EQUIRECT, folder, 'image_02'))
        self.files = list(filter(lambda x: x.endswith('.png'), self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image_02 = Image.open(os.path.join(DATA_2D_EQUIRECT, self.folder, 'image_02', file))
        image_03_path = os.path.join(DATA_2D_EQUIRECT, self.folder, 'image_03', file)
        if not os.path.exists(image_03_path):
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
            Image.fromarray(side).save(f'test{j}.png')

        return np.array(sides), int(file.split('.')[0])


def process_result(i, j, result, folder):
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


def main():
    for folder in os.listdir(DATA_2D_EQUIRECT):
        dataset = SidesDataset(folder)
        loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)

        for sides, frame in tqdm.tqdm(loader):
            if sides is None:
                continue
            images = list(sides.reshape(-1, 700, 700, 3))
            inputs = processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            class_queries_logits = outputs.class_queries_logits
            masks_queries_logits = outputs.masks_queries_logits

            results = processor.post_process_panoptic_segmentation(outputs, target_sizes=list(
                map(lambda x: x.shape[:2], images)))

            frame = frame.repeat_interleave(6)

            list(map(lambda x: process_result(x[0], x[1][0] % 6, x[1][1], folder),
                     zip(frame, enumerate(results))))


if __name__ == '__main__':
    main()
