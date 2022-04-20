import cv2
import utils
import numpy as np
from functools import partial


def nl_means_parameters(sigma: int, h: float, N: int, S: int):
    if (sigma > 0 and sigma <= 15):
        return 0.4 * sigma, 3, 21
    elif (sigma > 15 and sigma <= 30):
        return 0.4 * sigma, 5, 21
    elif (sigma > 30 and sigma <= 45):
        return 0.35 * sigma, 7, 21
    elif (sigma > 45 and sigma <= 75):
        return 0.35 * sigma, 9, 35
    elif (sigma > 45 and sigma <= 75):
        return 0.35 * sigma, 9, 35
    else:
        return 3.0, 7, 21


def denoise_gaussian_callback(
    val: int, img: np.ndarray, win_name: str, bilateral: bool
):
    assert len(img.shape) == 2, 'Multichannel images are not supported\n'

    gaussina_noise_sigma = \
        cv2.getTrackbarPos('gaussian noise sigma', win_name)
    noisy_img = img.copy()
    if (gaussina_noise_sigma > 0):
        gaussian_noise = \
            np.random.normal(scale=gaussina_noise_sigma, size=img.shape)
        noisy_img = img + gaussian_noise
        noisy_img = noisy_img.astype('uint8')

    img_denoised = img.copy()

    if bilateral:
        filter_size = cv2.getTrackbarPos('filter size', win_name)
        sigma_color = cv2.getTrackbarPos('sigma color', win_name)
        sigma_space = cv2.getTrackbarPos('sigma space', win_name)
        if filter_size > 0:
            cv2.bilateralFilter(
                noisy_img, filter_size, sigma_color, sigma_space, img_denoised
            )
        else:
            print(
                f'Cannot apply bilateral filtering: filter size {filter_size}'
                f'should be > 0'
            )
    else:
        sigma_nlmeans = cv2.getTrackbarPos('sigma nl means', win_name)
        if (sigma_nlmeans > 0):
            h, N, S = nl_means_parameters(sigma_nlmeans)
            cv2.fastNlMeansDenoising(noisy_img, img_denoised, h, N, S)
        else:
            print(
                f'Cannot apply non local filtering filtering: sigma'
                f' {sigma_nlmeans} should be > 0'
            )
    cv2.imshow('denoising', noisy_img)
    cv2.imshow('Denoised image', img_denoised)
    diff = cv2.absdiff(img_denoised, noisy_img)
    cv2.normalize(diff, diff, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('difference', diff)


def main():
    # Initial values for parameters:
    bilateral = True
    gaussian_noise_sigma = 0
    filter_size = 7
    sigma_color = 0
    sigma_space = 0
    sigma_nlmeans = 0
    win_name = 'denoising'

    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'lena.png'),
        cv2.IMREAD_GRAYSCALE
    )
    cv2.namedWindow(win_name)
    partial_callback = partial(
        denoise_gaussian_callback, img=img, win_name=win_name, bilateral=bilateral
    )
    cv2.createTrackbar(
        'gaussian noise sigma', 'denoising', gaussian_noise_sigma,
        50, partial_callback
    )
    if bilateral:
        cv2.createTrackbar(
            'filter size', 'denoising', filter_size, 50, partial_callback
        )
        cv2.createTrackbar(
            'sigma color', 'denoising', sigma_color, 200, partial_callback
        )
        cv2.createTrackbar(
            'sigma space', 'denoising', sigma_space, 100, partial_callback
        )
    else:
        cv2.createTrackbar(
            'sigma nl means', 'denoising', sigma_nlmeans, 100, partial_callback
        )
    denoise_gaussian_callback(0, img, win_name, bilateral)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
