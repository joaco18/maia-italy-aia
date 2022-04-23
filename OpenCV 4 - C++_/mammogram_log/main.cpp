#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
    cv::Mat img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram.tif",
        cv::IMREAD_UNCHANGED
    );
    aia::imshow("Original Image histogram", ucas::imhist(img));
    cv::Mat img_8U = img.clone();
    double Min,Max;
    cv::minMaxLoc(img_8U, &Min, &Max);
    img_8U -= Min;
    img_8U.convertTo(img_8U, CV_8U, 255.0/(Max-Min));
    aia::imshow("Mammogram", img_8U, true, 0.2);

    // Get the bitdepth of the image
    int bpp = ucas::imdepth_detect(img);
    int L = std::pow(2, bpp);
    printf("bpp = %d, L = %d\n", bpp, L);

    // Apply the transformation
    // Get the normalization factor to turn intensity range to [0, L-1]
    double c = (L - 1) / std::log(L); 
    for (int y = 0; y < img.rows; y++)
    {
        unsigned short* yRow = img.ptr<unsigned short>(y);
        for (int x = 0; x < img.cols; x++)
        {
            yRow[x] = c * std::log(1 + yRow[x]);
        }
    }
    // Invert intensity range (dicom -1)
    img = (L - 1) - img;
    cv::imwrite(
        std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram_processed.tif",
        img
    );
    aia::imshow("Normalized Image histogram", ucas::imhist(img));
    img_8U = img.clone();
    cv::minMaxLoc(img_8U, &Min, &Max);
    img_8U -= Min;
    img_8U.convertTo(img_8U, CV_8U, 255.0/(Max-Min));
    aia::imshow("Mammogram", img_8U, true, 0.2);
    return EXIT_SUCCESS;
}
