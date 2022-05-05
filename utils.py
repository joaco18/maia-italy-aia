import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from pathlib import Path
from typing import Callable

EXAMPLES_DIR = Path("../project configuration/OpenCV 4 - C++_/example-images")
CAMERA_FPS = 10


def bitdepth(ocv_depth: str):
    if ocv_depth == 'uint8':
        return 8
    elif ocv_depth == 'int8':
        return 8
    elif ocv_depth == 'uint16':
        return 16
    elif ocv_depth == 'int16':
        return 16
    elif ocv_depth == 'int32':
        return 32
    elif ocv_depth == 'float32':
        return 32
    elif ocv_depth == 'float64':
        return 64
    else:
        return -1


def cv2_imshow(title: str, img: np.ndarray, cmap: str = None):
    plt.figure()
    plt.title(title)
    if cmap is not None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()


def hist_plot(values: np.ndarray, freqs: np.ndarray, title: str = 'Histogram'):
    plt.figure()
    plt.title(title)
    plt.ylabel('Freq')
    plt.xlabel('Intensity')
    plt.plot(values[:-1], freqs)
    plt.fill_between(
        x=values[:-1], y1=freqs, interpolate=True, color='blue', alpha=0.1
    )
    plt.show()


def imdepth_detect(img: np.ndarray):
    return np.ceil(np.log2(img.max()))


@jit(nopython=True)
def our_hist_numba(img):
    freqs = np.zeros((255,))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            freqs[img[y, x]] += 1
    return freqs


def process_video_stream(
    inputPath: str = "",
    outputPath: str = "",
    show_only_processed_stream: bool = False,
    delay: int = 0,
    frame_processor: Callable = None,
):
    """
    Args:
        frame_processor (): frame-by-frame image processing function
        inputPath (str, optional): input video path (if empty,
            camera will be used). Defaults to "".
        outputPath (str, optional): output video path (optional).
            Defaults to "".
        show_only_processed_stream (bool, optional): Defaults to False.
        delay (int, optional): Defaults to 0.
    """
    # Open input video stream (either from camera or from file)
    if (inputPath != ""):
        capture = cv2.VideoCapture(inputPath)
    else:
        capture = cv2.VideoCapture(0)
    assert capture.isOpened(), \
        f'Cannot open input video stream from ' \
        f'{"camera" if (inputPath == "") else inputPath}'

    # get frame rate (also known as Frames Per Second)
    fps = CAMERA_FPS if (inputPath == "") else capture.get(cv2.CAP_PROP_FPS)

    # open output stream (if required)
    if (outputPath != ""):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = cv2.VideoWriter(outputPath, fourcc, fps, (f_w, f_h), True)
        assert output.isOpened(), 'Cannot open output video stream'

    # for all frames in video
    stop = False
    print(int(1000/fps))
    if delay == 0:
        delay = 1 if inputPath == "" else int(1000/fps)
    while not stop:
        # read next frame if any
        ret, frame = capture.read()

        assert ret, "Couldn't load the frame"

        # process frame
        processed_frame = frame_processor(frame) \
            if (frame_processor is not None) else frame.copy()

        # display original and processed frame
        if not show_only_processed_stream:
            cv2.imshow("Original stream", frame)

        # cv2.resize(processed_frame, processed_frame, (0, 0), 0.5, 0.5)
        cv2.imshow("Processed stream", processed_frame)

        # write output frame
        if (outputPath != "") and output.isOpened():
            output.write(processed_frame)

        # introduce a delay
        if (cv2.waitKey(delay) >= 0):
            stop = True
            cv2.destroyWindow("Processed stream")
    capture.release()
    if (outputPath != "") and output.isOpened():
        output.release()


def get_triangle_auto_threshold(histogram: np.ndarray):
    # find min and max
    idxs = np.nonzero(histogram)[0]

    minv = idxs.min()
    maxv = idxs.max()
    minv = minv - 1 if minv > 0 else minv
    # The Triangle algorithm cannot tell whether the data is skewed
    # to one side or another. This causes a problem as there are 2
    # possible thresholds between the max and the 2 extremes of the
    # histogram. Find out to which side of the max point the data is
    # furthest, and use that as the other extreme.
    minv2 = maxv + 1 if maxv < histogram.size else maxv
    maxv = np.argmax(histogram)
    print(f'{minv} {maxv} {minv2}')
    inverted = False
    if (maxv - minv) < (minv2 - maxv):
        # reverse the histogram
        print("Reversing histogram.")
        inverted = True
        histogram = np.flip(histogram)
        minv = histogram.size - 1 - minv2
        maxv = histogram.size - 1 - maxv

    if minv == maxv:
        print("Triangle:  min == max.")
        return minv

    # describe line by nx * x + ny * y - d = 0
    # nx is just the max frequency as the other point has freq=0
    nx = histogram[maxv]
    ny = minv - maxv
    d = np.sqrt(nx*nx + ny*ny)
    nx /= d
    ny /= d
    d = nx * minv + ny * histogram[minv]

    # find the split point
    split = minv
    split_distance = 0
    for i in range(minv + 1, maxv + 1):
        new_distance = nx * i + ny * histogram[i] - d
        if new_distance > split_distance:
            split = i
            split_distance = new_distance
    split -= 1

    # reverse back the histogram
    if inverted:
        histogram = np.flip(histogram)
        return histogram.size - 1 - split
    else:
        return split


def apply_hue_offset(hue: float, offset: float):
    if hue < offset:
        return offset - hue
    else:
        return hue - offset