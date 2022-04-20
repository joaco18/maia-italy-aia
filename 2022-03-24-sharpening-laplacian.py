import cv2
import utils
import numpy as np
from functools import partial


def sharpen_laplacian_callback(
    k_x10: int, img: np.ndarray, win_name: str
):
    k = k_x10 / 10.
    kernel = np.array([
        [-k,     -k,     -k],
        [-k,    1+8*k,   -k],
        [-k,     -k,     -k],
    ])
    img_sharpened = cv2.filter2D(img, cv2.CV_8U, kernel)

    cv2.imshow(win_name, img_sharpened)


def main():
    win_name = "Sharpening"
    k_x10 = 0

    img = cv2.imread(str(utils.EXAMPLES_DIR/'eye.blurry.png'))
    cv2.resize(img, (-1, -1), img, 2, 2)

    cv2.namedWindow(win_name)
    partial_callback = partial(
        sharpen_laplacian_callback, img=img, win_name=win_name
    )
    cv2.createTrackbar("k", win_name, k_x10, 100, partial_callback)
    sharpen_laplacian_callback(k_x10, img, win_name)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
