#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
    cv::Mat input_img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/tools.png",
        cv::IMREAD_GRAYSCALE
    );
    aia::imshow("Original image", input_img);
    aia::imshow("Histogram", ucas::imhist(input_img));

    // Otsu method
    int T = ucas::getOtsuAutoThreshold(ucas::histogram(input_img));
    printf("Otsu T = %d\n", T);
    cv::Mat binarized_img;
    cv::threshold(input_img, binarized_img, T, 255, cv::THRESH_BINARY);
    aia::imshow("Otsu-binarized imagee", binarized_img);

    // Triangle method
    T = ucas::getTriangleAutoThreshold(ucas::histogram(input_img));
    printf("Triangular T = %d\n", T);
    cv::threshold(input_img, binarized_img, T, 255, cv::THRESH_BINARY);
    aia:imshow("Triangle-binarized image", binarized_img);

    cv::waitKey(0);
    return EXIT_SUCCESS;
}