import cv2
import utils


def main():
    input_img = cv2.imread(str(utils.EXAMPLES_DIR/'retina.png'))

    result = cv2.pyrMeanShiftFiltering(input_img, 2, 30, 0)

    cv2.imshow('Original', input_img)
    cv2.waitKey(0)
    cv2.imshow('Mean Shift result', result)
    cv2.waitKey(0)

    cv2.imwrite(str(utils.EXAMPLES_DIR/'retina.MS.png'), result)


if __name__ == '__main__':
    main()
