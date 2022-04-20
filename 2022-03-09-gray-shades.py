import numpy as np
import time
import utils

""" Generate a gradient image of grayscales """


def main():
    start = time.time()
    [rows, cols] = (512, 512)
    column = (np.arange(0, rows)/(rows-1))*255
    img = np.repeat(column.astype('uint8').T.reshape((-1, 1)), cols, axis=1)
    print(f'Elapsed time (slow version) = {time.time()-start:.3f}')
    utils.cv2_imshow("Image", img, 'gray')


if __name__ == '__main__':
    main()
