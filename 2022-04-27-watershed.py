import cv2
import utils
import numpy as np


def main():
    img = cv2.imread(str(utils.EXAMPLES_DIR/'coins2.jpg'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, ker)
    cv2.imshow('Binarized', binarized)
    cv2.waitKey(0)

    # TODO: Solve this bug dist_transform is empty
    dist_transform = cv2.distanceTransform(
        binarized.astype('uint8'), cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(dist_transform, 0, 255, cv2.NORM_MINMAX)
    dist_transform = dist_transform.astype('uint8')
    cv2.imshow('Distance transform', dist_transform)
    cv2.waitKey(0)

    _, dist_transform_bin = cv2.threshold(
        dist_transform, 0.6*255, 255, cv2.THRESH_BINARY)
    cv2.imshow('Distance transform threshold', dist_transform_bin)
    cv2.waitKey(0)

    objects = cv2.findContours(
        dist_transform_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Internal markers
    markers = np.zeros_like(img, dtype='uint8')
    for i in range(len(objects)):
        cv2.drawContours(markers, objects, i, i+1, cv2.FILLED)

    # External markers
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    external_marker_mask = cv2.erode(255 - binarized, ker)
    markers = np.where(external_marker_mask, len(objects), 0)

    markers_vis = cv2.normalize(markers, 0, 255, cv2.NORM_MINMAX)
    markers_vis = markers_vis.astype('uint8')

    cv2.watershed(img, markers)
    #  -1 = dams				--> *255 = -255 --> +255 = 0 --> CV_8U = 0
    #  0 = not present			--> ...			--> ...		 --> ...
    #  [1, max index] = objects	--> >= 255		--> >= 255	 --> 255

    markers = (markers * 255 + 255).astype('uint8')
    markers = 255 - markers

    objects = cv2.findContours(markers, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in range(len(objects)):
        cv2.drawContours(img, objects, i, (0, 0, 255), 2)
    cv2.imshow('Segmented image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
