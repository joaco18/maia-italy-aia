import cv2
import utils

def main():
    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'brain_ct.jpeg'),
        cv2.IMREAD_GRAYSCALE
    )
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    
    result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)
    cv2.imshow('Gradient Image', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
