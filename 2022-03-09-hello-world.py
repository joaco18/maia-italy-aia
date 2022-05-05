import cv2
import utils
from pathlib import Path

# EXAMPLE 1: Load and show an RGB image
# EXAMPLE 2: load and show a grayscale image (8 bits per pixel)
# EXAMPLE 3: load and show a grayscale image (16 bits per pixel)
# EXAMPLE 4: do some image processing and save the result
# EXAMPLE 5: video processing: Face Detection

examples_folder = Path("../project configuration/OpenCV 4 - C++/example_images")


def main():
    # ////////////////////////// EXAMPLE 1 ///////////////////////////
    imgRGB = cv2.imread(str(examples_folder/'lena.png'), cv2.IMREAD_UNCHANGED)
    assert imgRGB is not None, "Cannot load image"
    print(
        f'Image loaded: dims = {imgRGB.shape[0]}, columns = {imgRGB.shape[1]},'
        f' channels = {imgRGB.shape[2]}, bits = {utils.bitdepth(imgRGB.dtype)}'
    )
    utils.cv2_imshow("An RGB image", imgRGB)

    # ////////////////////////// EXAMPLE 2 ///////////////////////////
    imgGray8 = cv2.imread(str(examples_folder/'lowcontrast.png'), cv2.IMREAD_UNCHANGED)
    assert imgGray8 is not None, "Cannot load image"
    print(
        f'Image loaded: dims = {imgGray8.shape[0]}, columns = {imgGray8.shape[1]},'
        f' channels = 1, bits = {utils.bitdepth(imgGray8.dtype)}'
    )
    utils.cv2_imshow("An 8-bits gray scale image", imgGray8, 'gray')

    # ////////////////////////// EXAMPLE 3 ///////////////////////////
    imgGray16 = cv2.imread(str(examples_folder/'raw_mammogram.tif'), cv2.IMREAD_UNCHANGED)
    assert imgGray16 is not None, "Cannot load image"
    print(
        f'Image loaded: dims = {imgGray16.shape[0]}, columns = {imgGray16.shape[1]},'
        f' channels = 1, bits = {utils.bitdepth(imgGray16.dtype)}'
    )
    utils.cv2_imshow("An 8-bits gray scale image", imgGray16, 'gray')

    # ////////////////////////// EXAMPLE 4 ///////////////////////////
    utils.cv2_imshow("Original", imgGray8, 'gray')
    eqImgGray8 = cv2.equalizeHist(imgGray8)
    cv2.imwrite(str(examples_folder/'lowconstrast_eq.png'), eqImgGray8)
    utils.cv2_imshow("Equalized", eqImgGray8, 'gray')


if __name__ == '__main__':
    main()
