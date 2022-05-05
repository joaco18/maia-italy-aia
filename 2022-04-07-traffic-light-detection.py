import cv2
import utils
import numpy as np


# ///////////// Parameters /////////////
A_min = 50
width_max_perc = 0.05
C_min = 0.65
S_min = 160


def avg_val_in_contour(img: np.ndarray, object_: np.ndarray):
    temp = np.zeros_like(img)
    cv2.drawContours(temp, object_, 0, (255, 255, 255), -1)
    return np.mean(img[temp == 255])


def frame_processor(img: np.ndarray):
    out_img = img.copy()

    # Step 1: enhance bright objects
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, binarized_v = cv2.threshold(
        img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 2: get connected components
    all_objects, _ = cv2.findContours(
        binarized_v, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    # Step 3: get circular objects only
    A_max = np.power(0.5 * (width_max_perc * img.shape[0]), 2) * np.pi
    for obj in all_objects:
        # Area criterium
        A = cv2.contourArea(obj)
        if (A < A_min) or (A > A_max):
            continue

        # circularity criterium
        p = cv2.arcLength(obj, True)
        C = (4 * np.pi * A) / (p * p)
        if C < C_min:
            continue

        # Color criterium
        avg_S = avg_val_in_contour(img_hsv[:, :, 1], obj)
        if avg_S < S_min:
            continue
        cv2.drawContours(out_img, obj, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return out_img


def main():
    utils.process_video_stream(
        str(utils.EXAMPLES_DIR/'traffic_light_4.mp4'),
        "", True, frame_processor=frame_processor
    )


if __name__ == '__main__':
    main()
