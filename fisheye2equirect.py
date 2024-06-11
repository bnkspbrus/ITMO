import cv2
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from kitti360scripts.helpers.project import CameraFisheye, CameraPerspective
from torch.utils.data import Dataset, DataLoader
import logging as log
import torch
from kitti360_config import CACHING_ENABLED


class CameraFisheyeV2(CameraFisheye):
    def cam2image(self, points):
        points = points.T
        norm = np.linalg.norm(points, axis=1)

        x = points[:, 0] / norm
        y = points[:, 1] / norm
        z = points[:, 2] / norm

        x /= z + self.fi['mirror_parameters']['xi']
        y /= z + self.fi['mirror_parameters']['xi']

        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']
        gamma1 = self.fi['projection_parameters']['gamma1']
        gamma2 = self.fi['projection_parameters']['gamma2']
        u0 = self.fi['projection_parameters']['u0']
        v0 = self.fi['projection_parameters']['v0']

        ro2 = x * x + y * y
        x *= 1 + k1 * ro2 + k2 * ro2 * ro2
        y *= 1 + k1 * ro2 + k2 * ro2 * ro2

        x = gamma1 * x + u0
        y = gamma2 * y + v0

        return x, y, norm * points[:, 2] / np.abs(points[:, 2])


def image2equirect(srcFrame, cam, outShape=(1400, 2800)):
    Hd, Wd = outShape
    i, j = np.meshgrid(np.arange(0, int(Hd)),
                       np.arange(0, int(Wd)))

    longitude = (j * 2.0 / Wd - 1) * np.pi
    latitude = (i * 2.0 / Hd - 1) * np.pi / 2

    x = (
            np.cos(latitude)
            * np.cos(longitude)
    )
    y = (
            np.cos(latitude)
            * np.sin(longitude)
    )
    z = np.sin(latitude)

    points_local = np.array([y.flatten(), z.flatten(), x.flatten()])

    map_x, map_y, depth = cam.cam2image(points_local)
    map_x[depth < 0] = -1
    map_y[depth < 0] = -1
    map_x = map_x.reshape(x.shape).T.astype(np.float32)
    map_y = map_y.reshape(x.shape).T.astype(np.float32)

    return cv2.remap(
        srcFrame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


KITTI_360 = '/home/jupyter/datasphere/project/KITTI-360/kitti360mm/raw'
KITTI_DATA_2D_RAW = 'data_2d_raw'
DATA_2D_EQUIRECT = '/home/jupyter/datasphere/project/ITMO/data_2d_equirect'
cam0 = CameraPerspective(KITTI_360, seq='2013_05_28_drive_0000_sync', cam_id=0)
cam1 = CameraPerspective(KITTI_360, seq='2013_05_28_drive_0000_sync', cam_id=1)
cam2 = CameraFisheyeV2(KITTI_360, seq='2013_05_28_drive_0000_sync', cam_id=2)
cam3 = CameraFisheyeV2(KITTI_360, seq='2013_05_28_drive_0000_sync', cam_id=3)


def process_image(seq, cam_id, image_name):
    image_path = osp.join(KITTI_360, KITTI_DATA_2D_RAW, seq, f'image_0{cam_id}', 'data_rgb', image_name)
    image = cv2.imread(image_path)
    if cam_id == 0:
        equi = image2equirect(image, cam0)
    elif cam_id == 1:
        equi = image2equirect(image, cam1)
    elif cam_id == 2:
        equi = image2equirect(image, cam2)
    elif cam_id == 3:
        equi = image2equirect(image, cam3)
    else:
        raise ValueError('Invalid cam_id')
    if CACHING_ENABLED:
        out_path = osp.join(DATA_2D_EQUIRECT, seq, f'image_0{cam_id}', image_name)
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, equi)
    return equi


class ImageDataset(Dataset):
    def __init__(self, sequence, cam_id, image_lower_bound, image_upper_bound):
        self.sequence = sequence
        self.cam_id = cam_id
        self.images = os.listdir(osp.join(KITTI_360, KITTI_DATA_2D_RAW, sequence, f'image_0{cam_id}', 'data_rgb'))
        self.images = list(filter(lambda x: x.endswith('.png'), self.images))
        self.images = list(
            filter(lambda x: image_lower_bound <= int(osp.splitext(x)[0]) <= image_upper_bound, self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if osp.exists(osp.join(DATA_2D_EQUIRECT, self.sequence, f'image_0{self.cam_id}', self.images[idx])):
            # load cached image
            return cv2.imread(osp.join(DATA_2D_EQUIRECT, self.sequence, f'image_0{self.cam_id}', self.images[idx])), int(
                osp.splitext(self.images[idx])[0])
        # process image
        return process_image(self.sequence, self.cam_id, self.images[idx]), int(osp.splitext(self.images[idx])[0])


def process_sequence(sequence, min_frame, max_frame):
    log.debug(f"Processing equirectangular projections: {sequence}/{min_frame}_{max_frame}")
    dataset = ImageDataset(sequence, 2, min_frame, max_frame)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    data_02 = list(tqdm(dataloader, total=len(dataloader)))
    rect_02 = torch.cat([torch.tensor(x) for x, _ in data_02], dim=0)
    frame_02 = torch.cat([torch.tensor(x) for _, x in data_02], dim=0)
    dataset = ImageDataset(sequence, 3, min_frame, max_frame)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    data_03 = list(tqdm(dataloader, total=len(dataloader)))
    rect_03 = torch.cat([torch.tensor(x) for x, _ in data_03], dim=0)
    frame_03 = torch.cat([torch.tensor(x) for _, x in data_03], dim=0)
    assert torch.all(frame_02 == frame_03)
    log.debug(f"Processing equirectangular projections finished: {sequence}/{min_frame}_{max_frame}")
    return rect_02, rect_03, frame_02


if __name__ == '__main__':
    process_sequence('2013_05_28_drive_0000_sync', 2, 385)
