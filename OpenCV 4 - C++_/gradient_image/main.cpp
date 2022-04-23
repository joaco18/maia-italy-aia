#include "aiaConfig.h"
#include "ucasConfig.h"

cv::Mat frameProcessor(const cv::Mat & img)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat dx, dy;
    cv::Sobel(img_gray, dx, CV_32F, 1, 0);
    cv::Sobel(img_gray, dy, CV_32F, 0, 1);
    
    cv::Mat mag;
    cv::magnitude(dx, dy, mag);

    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    mag.convertTo(mag, CV_8U);

    return 2 * mag;
    
    // // Optional implementation of cartoonization
    // cv::Mat median_filtered;
    // cv::medianBlur(img, median_filtered, 7);

    // cv::Mat mask;
    // cv::threshold(3*mag, mask, 40, 255, cv::THRESH_BINARY);

    // median_filtered.setTo(cv::Scalar(0, 0, 0), mask);
    // return median_filtered;
}
 
int main()
{
    aia::processVideoStream("", frameProcessor);
    return EXIT_SUCCESS;
}