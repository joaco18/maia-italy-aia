#include "aiaConfig.h"
#include "ucasConfig.h"

namespace
{
    cv::Mat img;
    cv::Mat out;
    std::string win_name = "Gamma correction";
    int gamma_x10 = 10;

    static void gammaCorrectionCallback(int, void*)
    {
        double gamma = gamma_x10 / 10.0;     
            // I guess because the step in the slide bar
        double c = std::pow(255, 1 - gamma); // no sé por qué esto
        for (int y = 0; y < out.rows; y++)
        {
            unsigned char* yRowIn = img.ptr<unsigned char>(y);
            unsigned char* yRowOut = out.ptr<unsigned char>(y);
            for (int x = 0; x < out.cols; x++)
            {
                yRowOut[x] = c * std::pow(yRowIn[x], gamma);
            }
        }
        cv::imshow(win_name, out);
    }
}

int main()
{
    img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg",
        cv::IMREAD_GRAYSCALE
    );
    if (!img.data)
		throw aia::error("Cannot load image");
    cv::resize(img, img, cv::Size(0,0), 0.5, 0.5);

    cv::namedWindow(win_name);
    cv::createTrackbar(
        "gamma", win_name, &gamma_x10, 100, gammaCorrectionCallback
    );
    
    out = img.clone();
    gammaCorrectionCallback(0,0);
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}