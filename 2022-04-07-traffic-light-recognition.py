import cv2
import utils
import numpy as np
from functools import partial


# ///////////// Parameters /////////////
A_min = 50             # lights assumed to be larger than 8 pixels in diameter
width_max_perc = 0.05  # lights assumed < than width_max_perc*image.width
C_min = 0.65           # circularity threshold
S_min = 100            # saturation threshold
V_min = 150            # value (in hsV color space) threshold
upper_perc = 0.9       # only objects lying in the upper upper_perc of the image
hue_offset = 0
# reference hue values
green_hue = utils.apply_hue_offset(hue_offset, 60) if hue_offset != 0 else 60
orange_hue = utils.apply_hue_offset(hue_offset, 20) if hue_offset != 0 else 14
red_hue = utils.apply_hue_offset(hue_offset, 0) if hue_offset != 0 else 3
g_min = 0.8            # goodness threshold (applied on spatial majority voting)
history_size = 15      # for time-voting (see below)
color_history = []     # contains last history_size detected frame colors


def color2scalar(color: str):
    if color == 'green':
        return (0, 255, 0)
    elif color == 'orange':
        return (0, 215, 215)
    elif color == 'red':
        return (0, 0, 255)
    else:
        return (128, 128, 128)


def color2text(color: str):
    if color == 'green':
        return 'GO'
    elif color == 'orange':
        return 'SLOW DOWN'
    elif color == 'red':
        return 'STOP'
    else:
        return '...'


def avg_val_in_contour(
    img: np.ndarray, obj: np.ndarray,
    hue_offset: int = None, cosine_correction: bool = False
):
    temp = np.zeros_like(img)
    cv2.drawContours(temp, obj, 0, (255, 255, 255), -1)
    if cosine_correction:
        img = np.rad2deg(np.arccos(np.cos(np.deg2rad(img * 2)))) / 2
    if hue_offset is not None:
        img = np.where(
            img < hue_offset, hue_offset - img, img - hue_offset
        )
    return np.mean(img[temp == 255])


def goodness(avgS: float, avgV: float, C: float):
    return (C + avgS / 255.0 + avgV / 255.0) / 3


def frame_processor(img: np.ndarray):
    out_img = img.copy()
    spatial_voting = {'unknown': 0, 'green': 0, 'orange': 0, 'red': 0}
    # Step 1: enhance bright objects
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, binarized_v = cv2.threshold(
        img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 2: get connected components
    all_objects, _ = cv2.findContours(
        binarized_v, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    # Step 3: get circular objects only
    A_max = np.power(0.5 * (width_max_perc * img.shape[0]), 2) * np.pi
    for obj in all_objects:
        # Object location criterium
        bbox = cv2.boundingRect(obj)
        # print(bbox)
        if (bbox[1] + bbox[3] / 2.0) > (upper_perc * img.shape[0]):
            continue

        # Area criterium
        A = cv2.contourArea(obj)
        if (A < A_min) or (A > A_max):
            continue

        # circularity criterium
        p = cv2.arcLength(obj, True)
        C = (4 * np.pi * A) / (p * p)
        if C < C_min:
            continue

        # Color criterium
        avg_S = avg_val_in_contour(img_hsv[:, :, 1], obj)
        if avg_S < S_min:
            continue

        # Luminosity:
        avg_V = avg_val_in_contour(img_hsv[:, :, 2], obj)
        if avg_V < V_min:
            continue

        # color recognition
        if hue_offset:
            avg_H = avg_val_in_contour(img_hsv[:, :, 1], obj, hue_offset=hue_offset)
        else:
            avg_H = avg_val_in_contour(
                img_hsv[:, :, 1], obj, cosine_correction=True
            )
        dist_green = np.abs(avg_H - green_hue)
        dist_orange = np.abs(avg_H - orange_hue)
        dist_red = np.abs(avg_H - red_hue)
        if (dist_green < dist_orange) and (dist_green < dist_red):
            light_color = 'green'
            # dist_H = dist_green
        elif (dist_orange < dist_green) and (dist_orange < dist_red):
            light_color = 'orange'
            # dist_H = dist_orange
        else:
            light_color = 'red'
            # dist_H = dist_red
        g = goodness(avg_S, avg_V, C)
        spatial_voting[light_color] += g

        cv_color = color2scalar(light_color)
        cv2.drawContours(out_img, obj, -1, cv_color, 3, cv2.LINE_AA)
        put_text = partial(
            cv2.putText, fontFace=2, fontScale=0.4, color=cv_color
        )
        br = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]])
        put_text(out_img, f'H = {avg_H}:.0f', br - np.array([-5, bbox[3]]))
        put_text(out_img, f'S = {avg_S}:.0f', br - np.array([-5, bbox[3] - 13]))
        put_text(out_img, f'V = {avg_V}:.0f', br - np.array([-5, bbox[3] - 26]))
        put_text(out_img, f'C = {C}:.0f', br - np.array([-5, bbox[3] - 39]))
        put_text(out_img, f'g = {g}:.0f', br - np.array([-5, bbox[3] - 52]))

        max_g = g_min
        frame_color = 'unknown'
        for key, val in spatial_voting.items():
            if val >= max_g:
                max_g = val
                frame_color = key

        color_history.append(frame_color)
        time_voting = {'unknown': 0, 'green': 0, 'orange': 0, 'red': 0}
        for color in color_history:
            time_voting[color] += 1

        max_freq = 0
        decision = 'unknown'
        for key, val in time_voting.items():
            if val >= max_freq:
                max_freq = val
                decision = key

        cv2.putText(
            out_img, decision, (100, 100), 2, 3, color2scalar(decision), 3
        )
    return out_img


def main():
    utils.process_video_stream(
        str(utils.EXAMPLES_DIR/'traffic_light_3.mp4'),
        "", True, frame_processor=frame_processor
    )


if __name__ == '__main__':
    main()
