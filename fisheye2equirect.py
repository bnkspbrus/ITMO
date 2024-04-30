import cv2
import numpy as np
import os
import tqdm
from kitti360scripts.helpers.project import CameraFisheye, CameraPerspective
from multiprocessing import Pool


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


KITTI_360 = 'KITTI-360'
DATA_2D_RAW = 'data_2d_raw'
DATA_2D_EQUIRECT = 'data_2d_equirect'
cam0 = CameraPerspective(KITTI_360, cam_id=0)
cam2 = CameraFisheyeV2(KITTI_360, cam_id=2)
cam3 = CameraFisheyeV2(KITTI_360, cam_id=3)


def process_image(args):
    image_name, seq, cam_id = args
    image_path = os.path.join(KITTI_360, DATA_2D_RAW, seq, f'image_0{cam_id}', 'data_rgb', image_name)
    image = cv2.imread(image_path)
    if cam_id == 0:
        equi = image2equirect(image, cam0)
    elif cam_id == 2:
        equi = image2equirect(image, cam2)
    elif cam_id == 3:
        equi = image2equirect(image, cam3)
    else:
        raise ValueError('Invalid cam_id')
    out_path = os.path.join(DATA_2D_EQUIRECT, seq, f'image_0{cam_id}', image_name)
    cv2.imwrite(out_path, equi)


def main():
    for folder in os.listdir(os.path.join(KITTI_360, DATA_2D_RAW)):
        for cam_id in range(4):
            images = os.path.join(KITTI_360, DATA_2D_RAW, folder, f'image_0{cam_id}', 'data_rgb')
            if not os.path.isdir(images):
                continue
            images = os.listdir(images)
            os.makedirs(os.path.join(DATA_2D_EQUIRECT, folder, f'image_0{cam_id}'), exist_ok=True)
            with Pool(4) as p:
                list(tqdm.tqdm(p.imap_unordered(process_image, ((image, folder, cam_id) for image in images)),
                               total=len(images)))


if __name__ == '__main__':
    main()
