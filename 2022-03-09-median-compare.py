import cv2
import time
import utils
from numba import jit
import numpy as np
from pathlib import Path

examples_folder = Path("../project configuration/OpenCV 4 - C++/example_images")


@jit(nopython=True)
def custom_median_filt(img, k, h):
    out = np.zeros((img.shape))
    for y in range(h, img.shape[0]-h):
        for x in range(h, img.shape[1]-h):
            out[y, x] = np.median(img[(y-h):(y+h+1), (x-h):(x+h+1)])
    return out


def main():
    img = cv2.imread(str(examples_folder/"lena.png"), cv2.IMREAD_GRAYSCALE)
    k = 7
    h = int(k/2)
    start = time.time()
    out = np.zeros((img.shape))
    for y in range(h, img.shape[0]-h):
        for x in range(h, img.shape[1]-h):
            out[y, x] = np.median(img[(y-h):(y+h+1), (x-h):(x+h+1)])
    print(f'Elapsed time (custom median filter) = {time.time()-start:.3f} seconds\n')
    utils.cv2_imshow("Image median-filtered (custom)", out, 'gray')

    start = time.time()
    out = custom_median_filt(img, k, h)
    print(f'Elapsed time (custom median filter numba first) = {time.time()-start:.3f} seconds\n')
    utils.cv2_imshow("Image median-filtered (custom numba first)", out, 'gray')

    start = time.time()
    out = custom_median_filt(img, k, h)
    print(f'Elapsed time (custom median filter numba second) = {time.time()-start:.3f} seconds\n')
    utils.cv2_imshow("Image median-filtered (custom numba second)", out, 'gray')

    start = time.time()
    dst = np.zeros((img.shape))
    dst = cv2.medianBlur(src=img, ksize=k)
    print(f'Elapsed time (OpenCV median filter) = {time.time()-start:.3f} seconds\n')
    utils.cv2_imshow("Image median-filtered (OpenCV)", dst, 'gray')


if __name__ == '__main__':
    main()
