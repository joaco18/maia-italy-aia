#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/photo/photo.hpp>

namespace
{
    cv::Mat img;
    
    ////// Parameters /////
    bool bilateral = true;
    int gaussian_noise_sigma = 0;
    
    // bilateral filter
    int filter_size = 7;
    int sigma_color = 0;
    int sigma_space = 0;

    // nonlocal means
    int sigma_nlmeans = 0;

    // Implement the table of parameters from the original paper
    void NlMeansParameters(int sigma, float& h, int& N, int& S)
    {
        h = 3.0f;
        N = 7;
        S = 21;
        if (sigma > 0 && sigma <= 15)
        {
            h = 0.40f * sigma;
            N = 3;
            S = 21;
        }
        if (sigma > 15 && sigma <= 30)
        {
            h = 0.40f * sigma;
            N = 5;
            S = 21;
        }
        if (sigma > 30 && sigma <= 45)
        {
            h = 0.35f * sigma;
            N = 7;
            S = 35;
        }
        if (sigma > 45 && sigma <= 75)
        {
            h = 0.35f * sigma;
            N = 9;
            S = 35;
        }
        if (sigma > 75 && sigma <= 100)
        {
            h = 0.30f * sigma;
            N = 11;
            S = 35;
        }
    }

    void denoiseGaussianCallback(int, void*)
    {
        if (img.channels() != 1)
        {
            std::cerr << "Multichannel images not supported\n";
            return;
        }
        // Add noise
        cv::Mat noisy_img = img.clone();
        if (gaussian_noise_sigma > 0)
        {
            cv::Mat gaussian_noise(img.rows, img.cols, CV_32F);
            cv::randn(
                gaussian_noise, cv::Scalar(0),
                cv::Scalar(gaussian_noise_sigma)
            );
            noisy_img.convertTo(noisy_img, CV_32F);
            noisy_img += gaussian_noise;
            noisy_img.convertTo(noisy_img, CV_8U);
        }

        // Denoise
        cv::Mat img_denoised = img.clone();
        if (bilateral)
        {
            if (filter_size > 0)
            {
                cv::bilateralFilter(
                    noisy_img, img_denoised, filter_size, sigma_color, sigma_space
                );
            }
            else
                printf("Cannot apply bilateral filtering: filter size (%d) should be > 0\n", filter_size);
        }
        else
        {
            if (sigma_nlmeans > 0)
            {
                float h = 0;
                int N = 0;
                int S = 0;
                NlMeansParameters(sigma_nlmeans, h, N, S);
                cv::fastNlMeansDenoising(noisy_img, img_denoised, h, N, S);
            }
            else
                printf("Cannot apply nonlocal means filtering: sigma (%d) should be > 0\n", sigma_nlmeans);
        }
        cv::imshow("denoising", noisy_img);
        cv::imshow("Denoised image", img_denoised);
        cv::Mat difference = cv::abs(img_denoised - noisy_img);
        cv::normalize(difference, difference, 0, 255, cv::NORM_MINMAX);
        cv::imshow("difference", difference);
    }   
}

int main()
{
    try
    {
        img = cv::imread(
            std::string(EXAMPLE_IMAGES_PATH) + "/lena.png",
            cv::IMREAD_GRAYSCALE
        );
        if (!img.data)
            throw ucas::Error("Cannot load image");
        
        bilateral = true;

        cv::namedWindow("denoising");
        cv::createTrackbar(
            "gaussian_noise", "denoising", &gaussian_noise_sigma,
            50, denoiseGaussianCallback
        );
        if (bilateral)
        {
            cv::createTrackbar(
                "filter_size", "denoising", &filter_size,
                50, denoiseGaussianCallback
            );
            cv::createTrackbar(
                "s_color", "denoising", &sigma_color,
                200, denoiseGaussianCallback
            );
            cv::createTrackbar(
                "s_space", "denoising", &sigma_space,
                100, denoiseGaussianCallback
            );
        }
        else
        {
			cv::createTrackbar(
                "s_nlmeans", "denoising",
                &sigma_nlmeans, 100, denoiseGaussianCallback
            );
        }
        denoiseGaussianCallback(0, 0);
        cv::waitKey(0);
		return EXIT_SUCCESS;
	}
	catch (aia::error& ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}
