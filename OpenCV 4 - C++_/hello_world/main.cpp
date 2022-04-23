#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// EXAMPLE 1: Load and show an RGB image
// EXAMPLE 2: load and show a grayscale image (8 bits per pixel)
// EXAMPLE 3: load and show a grayscale image (16 bits per pixel)
// EXAMPLE 4: do some image processing and save the result
// EXAMPLE 5: video processing: Face Detection

int main()
{
	try
	{
		////////////////////////// EXAMPLE 1 ///////////////////////////
		cv::Mat imgRGB = cv::imread(
			std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", cv::IMREAD_UNCHANGED
		);
		if (!imgRGB.data)
			throw aia::error("Cannor load image");
		
		printf(
			"Image loaded: dims = %d, columns = %d, channels = %d\n",
			imgRGB.rows, imgRGB.cols, imgRGB.channels(), aia::bitdepth(imgRGB.depth())
		);
		aia::imshow("An RGB image", imgRGB);

		////////////////////////// EXAMPLE 2 ///////////////////////////
		cv::Mat imgGray8 = cv::imread(
			std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png",
			cv::IMREAD_UNCHANGED
		);
		if (!imgGray8.data)
			throw aia::error("Cannor load image");
		
		printf(
		"Image loaded: dims = %d, columns = %d, channels = %d\n",
			imgGray8.rows, imgGray8.cols, imgGray8.channels(),
			aia::bitdepth(imgGray8.depth())
		);
		aia::imshow("An 8-bit gray scale image", imgGray8);

		////////////////////////// EXAMPLE 3 ///////////////////////////
		cv::Mat imgGray16 = cv::imread(
			std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram.tif",
			cv::IMREAD_UNCHANGED
		);
		if (!imgGray16.data)
			throw aia::error("Cannor load image");
		
		printf(
			"Image loaded: dims = %d, columns = %d, channels = %d\n",
			imgGray16.rows, imgGray16.cols, imgGray16.channels(),
			aia::bitdepth(imgGray16.depth())
		);

		cv::Mat imgGray16_8U;
		double Min,Max;
		cv::minMaxLoc(imgGray16, &Min, &Max);
		imgGray16 -= Min;
		imgGray16.convertTo(imgGray16_8U, CV_8U, 255.0/(Max-Min));

		aia::imshow("A 16-bit gray scale image", imgGray16_8U, true, 0.3f);

		////////////////////////// EXAMPLE 4 ///////////////////////////
		aia::imshow("Original", imgGray8);
		cv::Mat eqImgGray8(imgGray8.rows, imgGray8.cols, CV_8U, cv::Scalar(0));
		cv::equalizeHist(imgGray8, eqImgGray8);
		cv:imwrite(std::string(EXAMPLE_IMAGES_PATH)+ "/lowconstrast_eq.png", eqImgGray8);
		aia::imshow("Equalized", eqImgGray8);
		
		////////////////////////// EXAMPLE 5 ///////////////////////////
		aia::processVideoStream("", aia::project0::faceRectangles);

		return EXIT_SUCCESS;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}
