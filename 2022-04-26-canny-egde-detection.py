import cv2
import utils
import numpy as np
from functools import partial


def canny_edge_detection_callback(
    val, img: np.ndarray, win_name: str
):
    T_high = cv2.getTrackbarPos('T_high', win_name)
    sigma_x10 = cv2.getTrackbarPos('sigma', win_name)
    sigma = sigma_x10 / 10.
    k = int(6 * sigma)
    k = k + 1 if ((k % 2) == 0) else k
    img_blurred = cv2.GaussianBlur(img, (k, k), sigma, sigma)
    edge = cv2.Canny(img_blurred, T_high, T_high / 3)
    cv2.imshow(win_name, edge)


def main():
    win_name = "Gradient edges"
    sigma_x10 = 10
    T_high = 100

    img = cv2.imread(str(utils.EXAMPLES_DIR/'road.jpg'), cv2.IMREAD_GRAYSCALE)
    # cv2.resize(img, (-1, -1), img, 2, 2)

    cv2.namedWindow(win_name)
    partial_callback = partial(
        canny_edge_detection_callback, img=img, win_name=win_name
    )
    cv2.createTrackbar(
        "T_high", win_name,
        T_high, 300, partial_callback)
    cv2.createTrackbar(
        "sigma", win_name, sigma_x10, 100, partial_callback)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
