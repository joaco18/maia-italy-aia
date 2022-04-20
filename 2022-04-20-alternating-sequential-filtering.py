import cv2
import utils
import numpy as np


def main():
    img = cv2.imread(str(utils.EXAMPLES_DIR/'rice.png'), cv2.IMREAD_GRAYSCALE)
    img_copy = img.copy()

    i = 3
    while True:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

        cv2.imshow('Original image', img_copy)
        cv2.waitKey(0)
        cv2.imshow('ASF denoised', img)
        cv2.waitKey(0)
        cv2.imshow('Difference', np.abs(img_copy-img))
        cv2.waitKey(0)
        i += 2

        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    main()
