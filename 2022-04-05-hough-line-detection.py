import cv2
import utils
import numpy as np
from functools import partial


def edge_detection_grad_callback(
    val, img: np.ndarray, win_name: str
):
    stdevX10 = cv2.getTrackbarPos('stdev', win_name)
    threshold = cv2.getTrackbarPos('threshold', win_name)
    alpha1 = cv2.getTrackbarPos('alpha1', win_name)
    alpha0 = cv2.getTrackbarPos('alpha0', win_name)

    global img_edges

    if stdevX10 > 0:
        img_edges = cv2.GaussianBlur(img, (0, 0), stdevX10/10., stdevX10/10.)
    else:
        img_edges = img.copy()

    img_dx = cv2.Sobel(img_edges, cv2.CV_32F, 1, 0)
    img_dy = cv2.Sobel(img_edges, cv2.CV_32F, 0, 1)

    mag, angle = cv2.cartToPolar(img_dx, img_dy, angleInDegrees=True)

    temp = np.where(angle >= alpha0, 255, 0)
    temp2 = np.where(angle <= alpha1, 255, 0)
    temp = np.where((temp + temp2) != 0, 255, 0)
    temp2 = np.where(mag > threshold, 255, 0)
    img_edges = np.where((temp * temp2) != 0, 255, 0).astype('uint8')
    cv2.imshow(win_name, img_edges)


def hough_callback(val, img: np.ndarray, win_name: str):
    drho = cv2.getTrackbarPos('drho', win_name)
    dtheta = cv2.getTrackbarPos('dtheta', win_name)
    accum = cv2.getTrackbarPos('accum', win_name)
    n = cv2.getTrackbarPos('n', win_name)
    if drho <= 0:
        return
    if dtheta <= 0:
        return
    if accum <= 0:
        return

    img_copy = img.copy()
    lines = cv2.HoughLines(img_edges.astype('uint8'), drho, dtheta/180.0, accum)
    n = n if n < len(lines) else len(lines)
    for [[rho, theta]] in lines[:n]:
        if (theta < (np.pi / 4.)) or (theta > 3. * np.pi):
            pt1 = (int(rho / np.cos(theta)), 0)
            pt2 = (
                int(rho - img_copy.shape[0] * np.sin(theta) / np.cos(theta)),
                int(img_copy.shape[0])
            )
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 1)
        else:
            pt1 = (0, int(rho / np.sin(theta)))
            pt2 = (
                int(img_copy.shape[1]),
                int(rho - img_copy.shape[1] * np.cos(theta) / np.sin(theta))
            )
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(win_name, img_copy)


def main():
    stdevX10 = 10
    threshold = 60
    alpha0 = 0
    alpha1 = 360
    drho = 1
    dtheta = 1
    accum = 1
    n = 10

    img = cv2.imread(
        str(utils.EXAMPLES_DIR / 'road.jpg'),
        cv2.IMREAD_GRAYSCALE
    )

    grad_win_name = 'Edge detection (gradient)'
    cv2.namedWindow(grad_win_name)
    partial_grad_callback = partial(
        edge_detection_grad_callback, img=img, win_name=grad_win_name
    )
    cv2.createTrackbar(
        'stdev', grad_win_name, stdevX10, 100, partial_grad_callback
    )
    cv2.createTrackbar(
        'threshold', grad_win_name, threshold, 100, partial_grad_callback
    )
    cv2.createTrackbar(
        'alpha0', grad_win_name, alpha0, 360, partial_grad_callback
    )
    cv2.createTrackbar(
        'alpha1', grad_win_name, alpha1, 360, partial_grad_callback
    )
    edge_detection_grad_callback(0, img, grad_win_name)
    cv2.waitKey(0)

    hough_win_name = 'Edge detection (Hough)'
    partial_hough_callback = partial(
        hough_callback, img=img, win_name=hough_win_name
    )
    cv2.namedWindow(hough_win_name)
    cv2.createTrackbar('drho', hough_win_name, drho, 100, partial_hough_callback)
    cv2.createTrackbar('dtheta', hough_win_name, dtheta, 100, partial_hough_callback)
    cv2.createTrackbar('accum', hough_win_name, accum, 100, partial_hough_callback)
    cv2.createTrackbar('n', hough_win_name, n, 100, partial_hough_callback)
    hough_callback(0, img, hough_win_name)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
