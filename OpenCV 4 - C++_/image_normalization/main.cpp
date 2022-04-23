#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
    cv::Mat img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png",
        cv::IMREAD_GRAYSCALE
    );
    aia::imshow("Original Image", img);
    aia::imshow("Original Image Histogram", ucas::imhist(img));

    double maxv, minv;
    cv::minMaxLoc(img, &minv, &maxv);
    img = ((img - minv) / (maxv - minv)) * 255;

    // cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);

    aia::imshow("Image normalized", img);
    aia::imshow("Image normalized histogram", ucas::imhist(img));
    return EXIT_SUCCESS;
}