import cv2
import numpy as np
import os
import tqdm
from kitti360scripts.helpers.project import CameraFisheye
from multiprocessing import Pool


def dual_fisheye2equirect(srcFrame, cam1, cam2):
    inShape = srcFrame.shape[:2]
    outShape = srcFrame.shape[:2]
    Hs = inShape[0]
    Ws = inShape[1]
    Hd = outShape[0]
    Wd = outShape[1]

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

    vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T * 1000
    vertices1 = vertices[vertices[:, 1] < 0]
    vertices2 = vertices[vertices[:, 1] >= 0]

    cam2pose1 = cam1.camToPose
    cam2pose2 = cam2.camToPose

    points_local1 = cam1.world2cam(vertices1, cam2pose1[:3, :3], cam2pose1[:3, 3], True)
    points_local2 = cam2.world2cam(vertices2, cam2pose2[:3, :3], cam2pose2[:3, 3], True)

    map_x1, map_y1, depth1 = cam1.cam2image(points_local1)
    map_x2, map_y2, depth2 = cam2.cam2image(points_local2)

    map_x = np.zeros(Hd * Wd, dtype=np.float32)
    map_y = np.zeros(Hd * Wd, dtype=np.float32)
    map_x[vertices[:, 1] < 0] = map_x1
    map_y[vertices[:, 1] < 0] = map_y1
    map_x[vertices[:, 1] >= 0] = map_x2 + Ws // 2
    map_y[vertices[:, 1] >= 0] = map_y2

    map_x = map_x.reshape(x.shape).T.astype(np.float32)
    map_y = map_y.reshape(x.shape).T.astype(np.float32)

    return cv2.remap(
        srcFrame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


KITTI360_PATH = 'KITTI-360'
DATA_2D_RAW = 'KITTI-360/data_2d_raw'
DATA_2D_EQUIRECT = 'data_2d_equirect'
cam1 = CameraFisheye(KITTI360_PATH, cam_id=2)
cam2 = CameraFisheye(KITTI360_PATH, cam_id=3)


def process_image(folder, image_02, image_03):
    if image_02 != image_03:
        return
    img_02 = cv2.imread(os.path.join(DATA_2D_RAW, folder, 'image_02', 'data_rgb', image_02))
    img_03 = cv2.imread(os.path.join(DATA_2D_RAW, folder, 'image_03', 'data_rgb', image_03))
    srcFrame = np.hstack((img_02, img_03))
    equi = dual_fisheye2equirect(srcFrame, cam1, cam2)
    out_path = os.path.join(DATA_2D_EQUIRECT, os.path.basename(folder), os.path.basename(image_02))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, equi)


def main():
    for folder in os.listdir(DATA_2D_RAW):
        if not os.path.exists(os.path.join(DATA_2D_RAW, folder, 'image_02')):
            continue
        if not os.path.exists(os.path.join(DATA_2D_RAW, folder, 'image_03')):
            continue
        if not os.path.exists(os.path.join(DATA_2D_RAW, folder, 'image_02', 'data_rgb')):
            continue
        if not os.path.exists(os.path.join(DATA_2D_RAW, folder, 'image_03', 'data_rgb')):
            continue
        images_02 = os.listdir(os.path.join(DATA_2D_RAW, folder, 'image_02', 'data_rgb'))
        images_03 = os.listdir(os.path.join(DATA_2D_RAW, folder, 'image_03', 'data_rgb'))
        images_02 = [image for image in images_02 if image.endswith('.png')]
        images_03 = [image for image in images_03 if image.endswith('.png')]
        images_02 = sorted(images_02, key=lambda x: int(x.split('.')[0]))
        images_03 = sorted(images_03, key=lambda x: int(x.split('.')[0]))
        with Pool(8) as p:
            p.starmap(process_image, [(folder, image_02, image_03) for image_02, image_03 in zip(images_02, images_03)])


if __name__ == '__main__':
    main()
