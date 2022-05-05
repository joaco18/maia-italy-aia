import cv2
import utils
import numpy as np
from functools import partial


def object_picking_callback(
    event: int, x: int, y: int, flags, param, objects_filtered: np.ndarray,
    img: np.ndarray, win_name: str
):
    img_copy = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(len(objects_filtered)):
            test_res = cv2.pointPolygonTest(objects_filtered[i], (x, y), False)
            if test_res > 0:
                cv2.drawContours(
                    img_copy, objects_filtered, i, (255, 255, 255),
                    cv2.FILLED, cv2.LINE_AA
                )
            cv2.imshow(win_name, img_copy)


def main():
    win_name = "Object picking"
    img = cv2.imread(str(utils.EXAMPLES_DIR / 'tools.png'), cv2.IMREAD_GRAYSCALE)
    histogram = utils.our_hist_numba(img)
    T = utils.get_triangle_auto_threshold(histogram)
    _, binarized_img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    objects, _ = cv2.findContours(
        binarized_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    objects = np.asarray(objects)
    print(f'Objects (before filtering) = {len(objects)}')
    objects_filtered = []
    for i in range(len(objects)):
        if cv2.contourArea(objects[i]) > 100:
            objects_filtered.append(objects[i])
    print(f'Objects (after filtering) = {len(objects_filtered)}')

    cv2.namedWindow(win_name)
    partial_callback = partial(
        object_picking_callback, objects_filtered=objects_filtered,
        img=img, win_name=win_name
    )
    cv2.setMouseCallback(win_name, partial_callback)
    object_picking_callback(
        event=0, x=0, y=0, flags=0, param=0, objects_filtered=objects_filtered,
        img=img, win_name=win_name
    )
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
