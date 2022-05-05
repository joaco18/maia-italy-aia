import cv2
import utils
import numpy as np
from functools import partial


def unsharp_masking_callback(
    val, img: np.ndarray, win_name: str
):
    k_x10 = cv2.getTrackbarPos('k', win_name)
    k = k_x10 / 100.

    sigma_x10 = cv2.getTrackbarPos('sigma', win_name)
    sigma = sigma_x10 / 10.

    img_blurred = cv2.GaussianBlur(img, (-1, -1), sigmaX=sigma, sigmaY=sigma)
    img_sharpened = img + k * (img - img_blurred)
    img_sharpened = img_sharpened.astype('uint8')
    # cv2.normalize(img_sharpened, img_sharpened, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow(win_name, img_sharpened)


def main():
    win_name = "Sharpening"
    k_x10 = 0
    sigma_x10 = 10

    img = cv2.imread(str(utils.EXAMPLES_DIR/'eye.blurry.png'))
    cv2.resize(img, (-1, -1), img, 2, 2)

    cv2.namedWindow(win_name)
    partial_callback = partial(
        unsharp_masking_callback, img=img, win_name=win_name
    )
    cv2.createTrackbar("k", win_name, k_x10, 100, partial_callback)
    cv2.createTrackbar("sigma", win_name, sigma_x10, 100, partial_callback)
    unsharp_masking_callback(0, img, win_name)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
