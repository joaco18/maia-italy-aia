import cv2
import utils
import numpy as np


def main():
    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'raw_mammogram.tif'),
        cv2.IMREAD_UNCHANGED
    )

    utils.cv2_imshow('Original image', img, 'gray')
    freq, vals = np.histogram(img.flatten())
    utils.hist_plot(vals, freq)

    bpp = utils.imdepth_detect(img)
    L = 2 ** bpp

    print(f'bpp: {bpp}, L: {L}')

    # Get the normalization factor to turn intensity range to [0, L-1]
    c = (L - 1) / np.log(L)
    img = c * np.log(1 + img)
    img = (L - 1) - img

    cv2.imwrite(str(utils.EXAMPLES_DIR/'raw_mammogram_processed.tif'), img)

    utils.cv2_imshow('Original image', img, 'gray')
    freq, vals = np.histogram(img.flatten())
    utils.hist_plot(vals, freq)


if __name__ == '__main__':
    main()
