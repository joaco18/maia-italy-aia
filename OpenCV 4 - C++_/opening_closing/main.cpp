#include "aiaConfig.h"
#include "ucasConfig.h"


int main()
{
    cv::Mat input_img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH)+"/tools.png",
        cv::IMREAD_GRAYSCALE
    );
    
    aia::imshow("Original image", input_img);
    aia::imshow("Histogram", ucas::imhist(input_img));
    
    // OTSU
    int T = ucas::getOtsuAutoThreshold(ucas::histogram(input_img));
    printf("Otsu T = %d\n", T);
    
    cv::Mat binarized_img_otsu;
    cv::threshold(input_img, binarized_img_otsu, T, 255, cv::THRESH_BINARY);
    aia::imshow("Otsu binarized image", binarized_img_otsu);

    // TRIANGLE
    T = ucas::getTriangleAutoThreshold(ucas::histogram(input_img));
    printf("Triangle T = %d\n", T);
    
    cv::Mat binarized_img_triangle;
    cv::threshold(input_img, binarized_img_triangle, T, 255, cv::THRESH_BINARY);
    aia::imshow("Triangle binarized image", binarized_img_triangle);

    cv::Mat SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
    cv::morphologyEx(
        binarized_img_triangle, binarized_img_triangle, cv::MORPH_OPEN, SE
    );
    aia::imshow("Triangle binarized image after opening", binarized_img_otsu);
    cv::morphologyEx(
        binarized_img_otsu, binarized_img_otsu, cv::MORPH_CLOSE, SE
    );
    aia::imshow("Otsu binarized image after closing", binarized_img_otsu);

    cv::waitKey(0);

    return EXIT_SUCCESS;
}   
