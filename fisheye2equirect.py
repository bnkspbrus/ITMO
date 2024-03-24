import cv2
import numpy as np
import os
import glob
import tqdm

APERTURE = 180
RADIUS = 700
IN_SHAPE = [1400, 1400]
OUT_SHAPE = [700, 1400]


def fisheye2equirect(srcFrame,
                     outShape=OUT_SHAPE,
                     aperture=APERTURE,
                     delx=0,
                     dely=0,
                     radius=RADIUS):
    assert len(srcFrame.shape) == 3, "Input image must be a 3 channel image"
    assert srcFrame.shape[0] == IN_SHAPE[0] and srcFrame.shape[1] == IN_SHAPE[1], "Input image shape mismatch"

    Hs = IN_SHAPE[0]
    Ws = IN_SHAPE[1]
    Hd = outShape[0]
    Wd = outShape[1]

    Cx = (
            Ws // 2 - delx
    )
    Cy = (
            Hs // 2 - dely
    )

    i, j = np.meshgrid(np.arange(0, int(Hd)),
                       np.arange(0, int(Wd)))

    x = (
            radius
            * np.cos((i * 1.0 / Hd - 0.5) * np.pi)
            * np.cos((j * 1.0 / Hd - 0.5) * np.pi)
    )
    y = (
            radius
            * np.cos((i * 1.0 / Hd - 0.5) * np.pi)
            * np.sin((j * 1.0 / Hd - 0.5) * np.pi)
    )
    z = radius * np.sin((i * 1.0 / Hd - 0.5) * np.pi)

    r = (
            2
            * np.arctan2(np.sqrt(x ** 2 + z ** 2), y)
            / np.pi
            * 180
            / aperture
            * radius
    )
    theta = np.arctan2(z, x)

    map_x = np.multiply(r, np.cos(theta)).T.astype(np.float32) + Cx
    map_y = np.multiply(r, np.sin(theta)).T.astype(np.float32) + Cy

    return cv2.remap(
        srcFrame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def dual_fisheye2equirect(img1, img2):
    equi1 = fisheye2equirect(img1)
    equi2 = fisheye2equirect(img2)
    equi2 = np.roll(equi2, OUT_SHAPE[1] // 2, axis=1)
    equi = np.copy(equi2)
    equi[:, OUT_SHAPE[1] // 4:-(OUT_SHAPE[1] // 4)] = equi1[:, OUT_SHAPE[1] // 4:-(OUT_SHAPE[1] // 4)]
    return equi


DATA_2D_RAW = 'KITTI-360/data_2d_raw'
DATA_2D_EQUIRECT = './data_2d_equirect'

for folder in glob.glob(os.path.join(DATA_2D_RAW, '2013_05_28_drive_*_sync')):
    images_02 = glob.glob(os.path.join(folder, 'image_02/data_rgb/*.png'))
    images_03 = glob.glob(os.path.join(folder, 'image_03/data_rgb/*.png'))
    images_02 = sorted(images_02, key=lambda x: int(os.path.basename(x).split('.')[0]))
    images_03 = sorted(images_03, key=lambda x: int(os.path.basename(x).split('.')[0]))
    for image_02, image_03 in tqdm.tqdm(zip(images_02, images_03), total=len(images_02), desc=os.path.basename(folder)):
        assert os.path.basename(image_02) == os.path.basename(image_03), "Image names do not match for folder: {folder}"
        img_02 = cv2.imread(image_02)
        img_03 = cv2.imread(image_03)
        equi = dual_fisheye2equirect(img_02, img_03)
        out_path = os.path.join(DATA_2D_EQUIRECT, os.path.basename(folder), os.path.basename(image_02))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, equi)
