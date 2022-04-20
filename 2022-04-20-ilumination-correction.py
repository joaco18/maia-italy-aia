import cv2
import utils
import numpy as np


# TODO: NOT WORKING PROPERLY
def main():
    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'rice.png'),
        cv2.IMREAD_GRAYSCALE
    )

    _, bin = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imshow('Binarized on original', bin)
    cv2.waitKey(0)

    IF = cv2.morphologyEx(
        img, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    )
    cv2.imshow('IF', IF)
    cv2.waitKey(0)

    th = img - IF
    cv2.imshow('Tophat', th)
    cv2.waitKey(0)

    _, th = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Binarized after tophat', th)
    cv2.waitKey(0)

    avg_if = np.mean(IF)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.imshow('Corrected image', th + avg_if)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
