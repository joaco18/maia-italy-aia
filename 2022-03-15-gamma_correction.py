import cv2
import numpy as np
from pathlib import Path

examples_folder = Path("../project configuration/OpenCV 4 - C++/example_images")
win_name = 'Gamma correction'
gamma_x10 = 0


def main():
    img = cv2.imread(str(examples_folder/'lena.png'), cv2.IMREAD_GRAYSCALE)
    assert img is not None, 'Couldn\'t load the image'

    cv2.resize(src=img, dst=img, dsize=None, fx=0.5, fy=0.5)

    def gamma_correction_callback(gamma_x10):
        gamma = gamma_x10 / 10
        c = 255 ** (1 - gamma)
        out = c * np.power(img, gamma)
        cv2.imshow(win_name, out)

    cv2.namedWindow(win_name)
    cv2.createTrackbar(
        'gamma', win_name, 0, 100, gamma_correction_callback
    )
    gamma_correction_callback(0)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
