import cv2
import utils
import numpy as np


def frame_processor(img: np.ndarray):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(dx, dy)
    cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype('uint8')
    return 2 * mag
    # # Aditional "cartoonization"
    # median_filtered = cv2.medianBlur(img, 7)
    # _, mask = cv2.threshold(mag, 30, 255, cv2.THRESH_BINARY)
    # black = np.zeros(median_filtered.shape)
    # median_filtered[mask == 255, :] = black[mask == 255, :]
    # return median_filtered


def main():
    utils.process_video_stream(frame_processor=frame_processor)


if __name__ == '__main__':
    main()
