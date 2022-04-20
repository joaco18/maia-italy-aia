import cv2
import utils
import numpy as np


def main():
    img = cv2.imread(str(utils.EXAMPLES_DIR/'galaxy.jpg'))

    marker_cur = cv2.morphologyEx(
        img, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    )
    cv2.imshow('Marker', marker_cur)
    cv2.waitKey(0)

    mask = img.copy()
    marker_prev = np.zeros(img.shape)
    it = 0
    while np.count_nonzero(marker_cur - marker_prev):
        marker_prev = marker_cur.copy()
        marker_cur = cv2.dilate(
            marker_cur, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        marker_cur = np.minimum(marker_cur, mask)
        it += 1
        print(it)
    cv2.imshow('Reconstructed image', marker_cur)
    cv2.waitKey(0)
    cv2.imshow('Stars image', mask - marker_cur)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
