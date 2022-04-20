import cv2
import utils
import numpy as np


def main():
    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'text.png'),
        cv2.IMREAD_GRAYSCALE
    )
    cv2.imshow('Original image', img)
    cv2.waitKey(0)

    ret, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV
    )
    cv2.imwrite(str(utils.EXAMPLES_DIR/'text_binarized.png'), img)

    h_proj = np.sum(img, axis=1)
    heights_sum = 0
    rows_count = 0
    print(h_proj)
    for i in range(h_proj.size):
        if ((h_proj[i] != 0) and (h_proj[i-1] == 0)):
            row_start = i
        if ((h_proj[i] == 0) and (h_proj[i-1] != 0)):
            rows_count += 1
            heights_sum = heights_sum + i - row_start
    print(f'rows count = {rows_count}')
    avg_row_height = heights_sum / rows_count
    print(f'rows average height = {avg_row_height}')

    vertical_SE_height = 0.8 * avg_row_height - 2
    vertical_SE = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(vertical_SE_height))
    )
    img_eroded = cv2.erode(img, vertical_SE)
    cv2.imwrite(
        str(utils.EXAMPLES_DIR/'text_seeds.png'),
        img_eroded
    )

    marker_cur = img_eroded.copy()
    marker_prev = np.zeros(marker_cur.shape)
    mask = img.copy()
    while np.count_nonzero(marker_cur - marker_prev) > 0:
        marker_prev = marker_cur.copy()
        SE = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        marker_cur = cv2.dilate(marker_cur, SE)
        marker_cur[mask == 0] = 0
        cv2.imshow('Reconstruction in progress', marker_cur)
        cv2.waitKey(100)
    cv2.imshow('Result', marker_cur)
    cv2.waitKey(0)
    cv2.imwrite(
        str(utils.EXAMPLES_DIR/'text_long_chars.png'),
        marker_cur
    )


if __name__ == '__main__':
    main()
