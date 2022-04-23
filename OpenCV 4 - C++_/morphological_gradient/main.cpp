#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
    try
	{
		cv::Mat img = cv::imread(
            std::string(EXAMPLE_IMAGES_PATH) + "/brain_ct.jpeg",
            cv::IMREAD_GRAYSCALE
        );
		aia::imshow("Original image", img, true, 1.0);

		cv::Mat SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

		cv::morphologyEx(img, img, cv::MORPH_GRADIENT, SE);

		aia::imshow("Gradient image", img*3, true, 1.0);

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