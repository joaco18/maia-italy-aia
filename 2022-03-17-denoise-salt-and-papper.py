import cv2
import utils
import numpy as np
from functools import partial


def salt_and_pepper_denoise_callback(val, img: np.ndarray, win_name: str):
    img_corrupted = img.copy()
    salt_pepper_perc = \
        cv2.getTrackbarPos('salt and pepper percentage', win_name)
    median_filter_size = \
        cv2.getTrackbarPos('median filter size', win_name)

    number_of_pixels = int(salt_pepper_perc/100 * img.size)
    y_coords = np.random.randint(0, img.shape[0] - 1, number_of_pixels)
    x_coords = np.random.randint(0, img.shape[1] - 1, number_of_pixels)
    values = np.random.randint(0, 2, number_of_pixels) * 255

    img_corrupted[y_coords, x_coords] = values
    cv2.imshow(win_name, img_corrupted)
    if median_filter_size % 2:
        cv2.medianBlur(img_corrupted, median_filter_size, img_corrupted)
        cv2.imshow('Restored', img_corrupted)


def main():
    win_name = 'Salt-and-pepper demo'
    salt_and_pepper_perc = 10
    median_filter_size = 1

    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'lena.png'),
        cv2.IMREAD_GRAYSCALE
    )
    cv2.namedWindow(win_name)
    partial_callback = partial(
        salt_and_pepper_denoise_callback, img=img, win_name=win_name)
    cv2.createTrackbar(
        'salt and pepper percentage', win_name,
        salt_and_pepper_perc, 100, partial_callback
    )
    cv2.createTrackbar(
        'median filter size', win_name,
        median_filter_size, 50, partial_callback
    )
    salt_and_pepper_denoise_callback(0, img, win_name)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
