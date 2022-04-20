import cv2
import utils
from functools import partial


def clahe_callback(val, img, win_name):
    clip_limit = cv2.getTrackbarPos('clip_limit', win_name)
    tile_size = cv2.getTrackbarPos('tile_size', win_name)

    out = img.copy()
    cv2.imshow(win_name, img)
    if tile_size < 1:
        return

    clahe_filter = cv2.createCLAHE(clip_limit, (tile_size, tile_size))

    # BGR
    for i in range(3):
        out[:, :, i] = clahe_filter.apply(img[:, :, i])
    cv2.imshow("Result (BGR)", out)

    # HSV
    out = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out[:, :, 2] = clahe_filter.apply(out[:, :, 2])
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    cv2.imshow("Result (HSV)", out)

    # Lab
    out = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    out[:, :, 0] = clahe_filter.apply(out[:, :, 0])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    cv2.imshow("Result (Lab)", out)


def main():
    win_name = 'CLAHE'
    clip_limit = 10
    tile_size = 4

    img = cv2.imread(str(utils.EXAMPLES_DIR/'low-contrast-landscape.jpg'))
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.namedWindow(win_name)
    partial_callback = partial(clahe_callback, img=img, win_name=win_name)
    cv2.createTrackbar('tile_size', win_name, tile_size, 50, partial_callback)
    cv2.createTrackbar('clip_limit', win_name, clip_limit, 100, partial_callback)
    clahe_callback(0, img, win_name)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
