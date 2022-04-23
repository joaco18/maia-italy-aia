#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

// Equalization of a 8-bit grayscale image

cv::Mat histEqualize(cv::Mat img)
{
    if (img.channels() != 1)
    {
        std::cerr << "only one channel images are supported\n";
        return img;
    }

    // Histogram
    std::vector<float> hist(256);
    for (int y = 0; y < img.rows; y++)
    {
        unsigned char* yRow = img.ptr<unsigned char>(y);
        for (int x = 0; x < img.cols; x++)
        {
            hist[yRow[x]]++;
        }
    }
    // Normalized histogram
	int N = img.rows * img.cols;
	for (int i = 0; i < hist.size(); i++)
		hist[i] /= N;

    // CDF
    std::vector<float> cdf(256);
    cdf[0] = hist[0];
    for (int i = 1; i < hist.size(); i++)
        cdf[i] = cdf[i - 1] + hist[i];
    
    // Get the LUT for the equalization
    for (int i = 0; i < cdf.size(); i++)
        cdf[i] *= 255;

    // Perform equalization
    for (int y = 0; y < img.rows; y++)
    {
        unsigned char* yRow = img.ptr<unsigned char>(y);
        for (int x = 0; x < img.cols; x++)
            yRow[x] = cdf[yRow[x]];
    }
    return img;
}


int main()
{
    cv::Mat img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg",
        cv::IMREAD_GRAYSCALE
    );
    aia::imshow("Original Image", img);
    aia::imshow("Original Image Histogram", ucas::imhist(img));
    
    cv::Mat res;
    cv::normalize(img, res, 0, 255, cv::NORM_MINMAX);
    aia::imshow("Normalized Image", res);
    aia::imshow("Normalized Image Histogram", ucas::imhist(res));
    
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10);
    clahe->apply(img, res);
    aia::imshow("Clahe Enhaced Image", res);
    aia::imshow("Clahe Enhaced Image Histogram", ucas::imhist(res));

    res = histEqualize(img);
    aia::imshow("Equalized Image", res);
    aia::imshow("Equalized Image Histogram", ucas::imhist(res));
    
    return EXIT_SUCCESS;
}