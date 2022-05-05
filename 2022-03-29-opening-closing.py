import cv2
import utils
import numpy as np


def main():
    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'tools.png'),
        cv2.IMREAD_GRAYSCALE
    )
    utils.cv2_imshow('Original image', img, cmap='gray')
    freq = utils.our_hist_numba(img)
    utils.hist_plot(np.arange(256), freq)

    T, binarized_img_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f'Otsu threshold: {T}')
    utils.cv2_imshow('Otsu binarized image', binarized_img_otsu, 'gray')

    T = utils.get_triangle_auto_threshold(freq)
    print(f'Triangular threshold: {T}')
    _, binarized_img_triangular = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    utils.cv2_imshow(
        'Triangular binarized image', binarized_img_triangular, 'gray'
    )

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    cv2.morphologyEx(
        binarized_img_triangular, cv2.MORPH_OPEN, se, binarized_img_triangular
    )
    utils.cv2_imshow(
        'Triangular binarized image after opening',
        binarized_img_triangular, 'gray'
    )

    cv2.morphologyEx(binarized_img_otsu, cv2.MORPH_CLOSE, se, binarized_img_otsu)
    utils.cv2_imshow(
        'Otsu binarized image after closing', binarized_img_otsu, 'gray'
    )


if __name__ == "__main__":
    main()
