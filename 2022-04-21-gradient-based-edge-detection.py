import cv2
import utils
import numpy as np
from functools import partial


def gradient_edge_detection_callback(
    val, img: np.ndarray, win_name: str
):
    grad_mag_thresh = cv2.getTrackbarPos('grad_mag_thresh', win_name)

    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(dx, dy)

    _, mag = cv2.threshold(
        mag, (grad_mag_thresh / 100.) * mag.max(), 255, cv2.THRESH_BINARY, )
    mag = mag.astype('uint8')
    cv2.imshow(win_name, mag)


def main():
    win_name = "Gradient edges"
    grad_mag_thresh = 0

    img = cv2.imread(str(utils.EXAMPLES_DIR/'road.jpg'), cv2.IMREAD_GRAYSCALE)
    # cv2.resize(img, (-1, -1), img, 2, 2)

    cv2.GaussianBlur(img, (7, 7), 0, img, 0)

    cv2.namedWindow(win_name)
    partial_callback = partial(
        gradient_edge_detection_callback, img=img, win_name=win_name
    )
    cv2.createTrackbar("grad_mag_thresh",
                       win_name, grad_mag_thresh, 100, partial_callback)
    gradient_edge_detection_callback(0, img, win_name)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
