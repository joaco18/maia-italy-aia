import cv2
import numpy as np


def main():
    # Generate image
    img = np.zeros((800, 800), 'uint8')
    img = cv2.rectangle(img, (50, 50), (100, 150), 255, 2)
    img = cv2.rectangle(img, (400, 400), (200, 250), 255, 2)
    cv2.imshow('Image', img)
    cv2.waitKey(0)

    # Fill
    marker = np.zeros(img.shape)
    pt2 = (marker.shape[0] - 1, marker.shape[1] - 1)
    cv2.rectangle(marker, (0, 0), pt2, 255)
    mask = 255 - img
    cv2.imshow("Hole filling (mask)", mask)
    mask = np.where(mask != 0, 1, 0)
    cv2.waitKey(0)

    marker_prev = np.zeros(marker.shape)
    while (np.count_nonzero(marker - marker_prev) > 0):
        marker_prev = marker.copy()
        SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        marker = cv2.dilate(marker, SE)
        marker = marker * mask
        cv2.imshow("Hole filling (in progress)", marker)
        if cv2.waitKey(50) >= 0:
            cv2.destroyWindow("Hole filling (in progress)")
    cv2.imshow("Hole filling (result)", 255 - marker)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
