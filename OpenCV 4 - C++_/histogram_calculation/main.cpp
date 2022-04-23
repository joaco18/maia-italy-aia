#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
    cv::Mat img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png",
        cv::IMREAD_GRAYSCALE
    );

    // Unicas version based on cv::calcHist
    std::vector<int> histOV = ucas::histogram(img);

    // Custom simple based on uint8 image assumption
    std::vector<int> histOur(256);
    for (int y = 0; y < img.rows; y++)
    {
        unsigned char* yRow = img.ptr<unsigned char>(y);
        for (int x = 0; x < img.cols; x++)
        {
            histOur[yRow[x]]++;
        }
    }

    for (int i = 0; i < 256; i++)
    {
        printf("histOV[%d] = %d, histOur[%d] = %d\n", i, histOV[i], i, histOur[i]);
    }

    aia::imshow("Image histogram", ucas::imhist(img));
    return EXIT_SUCCESS;
}
