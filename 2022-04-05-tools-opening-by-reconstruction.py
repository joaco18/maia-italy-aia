import cv2
import utils
import numpy as np


def main():
    input_img = cv2.imread(
        str(utils.EXAMPLES_DIR/'tools.png'),
        cv2.IMREAD_GRAYSCALE
    )

    utils.cv2_imshow('Original image', input_img, 'gray')
    freq = utils.our_hist_numba(input_img)
    utils.hist_plot(np.arange(256), freq)

    T = utils.get_triangle_auto_threshold(freq)
    print(f'Triangle T = {T}')
    _, binarized_img_triangle = cv2.threshold(
        input_img, T, 255, cv2.THRESH_BINARY
    )
    utils.cv2_imshow('Triangle binarized image', binarized_img_triangle, 'gray')

    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    eroded_img = cv2.morphologyEx(binarized_img_triangle, cv2.MORPH_ERODE, SE)
    utils.cv2_imshow('Triangle after erosion', eroded_img, 'gray')

    marker_cur = eroded_img.copy()
    marker_prev = np.zeros(marker_cur.shape)
    mask = binarized_img_triangle.copy()
    while np.count_nonzero(marker_cur - marker_prev) > 0:
        marker_prev = marker_cur.copy()
        SE = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        marker_cur = cv2.dilate(marker_cur, SE)
        marker_cur[mask == 0] = 0
        cv2.imshow('Reconstruction in progress', marker_cur)
        cv2.waitKey(100)

    utils.cv2_imshow('Reconstruction result', marker_cur, 'gray')
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
