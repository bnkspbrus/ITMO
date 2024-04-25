import cv2
import numpy as np
import os
import tqdm
from kitti360scripts.helpers.project import CameraFisheye
from multiprocessing import Pool


def cam2image(cam, points):
    points = points.T
    norm = np.linalg.norm(points, axis=1)

    x = points[:, 0] / norm
    y = points[:, 1] / norm
    z = points[:, 2] / norm

    x /= z + cam.fi['mirror_parameters']['xi']
    y /= z + cam.fi['mirror_parameters']['xi']

    k1 = cam.fi['distortion_parameters']['k1']
    k2 = cam.fi['distortion_parameters']['k2']
    gamma1 = cam.fi['projection_parameters']['gamma1']
    gamma2 = cam.fi['projection_parameters']['gamma2']
    u0 = cam.fi['projection_parameters']['u0']
    v0 = cam.fi['projection_parameters']['v0']

    ro2 = x * x + y * y
    x *= 1 + k1 * ro2 + k2 * ro2 * ro2
    y *= 1 + k1 * ro2 + k2 * ro2 * ro2

    undistorted = np.array([x, y, np.ones_like(x)]).T
    undistorted = undistorted.reshape(-1, 1, 3)
    K = np.array([[gamma1, 0, u0], [0, gamma2, v0], [0, 0, 1]])
    points_image, _ = cv2.projectPoints(undistorted, np.zeros(3), np.zeros(3), K, None)
    x, y = points_image[:, 0, 0], points_image[:, 0, 1]

    return x, y, norm * points[:, 2] / np.abs(points[:, 2])


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

    map_x1, map_y1, depth1 = cam2image(cam1, points_local1)
    map_x2, map_y2, depth2 = cam2image(cam2, points_local2)

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
DATA_2D_RAW = 'data_2d_raw'
DATA_2D_EQUIRECT = 'data_2d_equi'
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
