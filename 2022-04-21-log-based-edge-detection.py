import cv2
import utils
import numpy as np
from numba import njit
from functools import partial

# TODO: Not working exactly as cpp version


@njit(cache=True)
def get_constrained_zero_crossings(
    img_log: np.ndarray, grad_mag_thresh: float, zero_cross_thresh: float
):
    output = np.zeros_like(img_log, dtype='uint8')
    candidate_points = np.where(img_log > grad_mag_thresh)
    for y, x in zip(candidate_points[0], candidate_points[1]):
        if x == 0 or x >= img_log.shape[1]-1 or y == 0 or y >= img_log.shape[0]-1:
            continue
        NW, N, NE, W, _, E, SW, S, SE = img_log[y - 1: y + 2, x - 1: x + 2].ravel()
        for a, b in [(N, S), (NE, SW), (E, W), (SE, NW)]:
            if ((a * b) > 0) and (np.abs(a - b) > zero_cross_thresh):
                output[y, x] = 255
                break
    return output


def improved_log_edge_detection_callback(
    val, img: np.ndarray, win_name: str
):
    # Get parameters
    grad_mag_thresh_perc = cv2.getTrackbarPos('grad_mag_thresh_perc', win_name)
    zero_cross_thresh_perc = cv2.getTrackbarPos('zero_cross_thresh_perc', win_name)
    sigma_x10 = cv2.getTrackbarPos('sigma', win_name)

    # Compute gradient
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(dx, dy)
    max_vg = mag.max()

    # LoG
    sigma = sigma_x10 / 10.
    k = int(6 * sigma)
    k = k + 1 if ((k % 2) == 0) else k

    img_log = cv2.GaussianBlur(img, (k, k), sigma, sigma)
    laplacian_kernel = np.array([
        [1,  1,  1],
        [1, -8,  1],
        [1,  1,  1]
    ], dtype=np.float32)
    img_log = cv2.filter2D(img_log, cv2.CV_32F, laplacian_kernel)

    # Get zero crossings
    max_v = img_log.max()
    zero_cross_thresh = (zero_cross_thresh_perc / 100.) * max_v
    grad_mag_thresh = (grad_mag_thresh_perc / 100.) * max_vg
    output = get_constrained_zero_crossings(
        img_log, grad_mag_thresh, zero_cross_thresh)
    cv2.imshow(win_name, output)


def main():
    img = cv2.imread(str(utils.EXAMPLES_DIR/'road.jpg'), cv2.IMREAD_GRAYSCALE)
    # cv2.resize(img, (-1, -1), img, 2, 2)

    win_name = "LoG Edge detection"
    sigma_x10 = 10
    grad_mag_thresh_perc = 10
    zero_cross_thresh_perc = 10

    cv2.namedWindow(win_name)
    partial_callback = partial(
        improved_log_edge_detection_callback, img=img, win_name=win_name
    )
    cv2.createTrackbar(
        "grad_mag_thresh_perc", win_name,
        grad_mag_thresh_perc, 100, partial_callback)
    cv2.createTrackbar(
        "sigma", win_name, sigma_x10, 100, partial_callback)
    cv2.createTrackbar(
        "zero_cross_thresh_perc", win_name,
        zero_cross_thresh_perc, 100, partial_callback)

    improved_log_edge_detection_callback(0, img, win_name)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
