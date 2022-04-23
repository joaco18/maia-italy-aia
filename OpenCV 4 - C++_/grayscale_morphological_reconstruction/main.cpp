#include "aiaConfig.h"
#include "ucasConfig.h"


int myCountNonZero(cv::Mat multichannel_img)
{
    cv::Mat channels[multichannel_img.channels()];
    cv::split(multichannel_img, channels);
    int count = 0;
    for (int i = 0; i < multichannel_img.channels(); i++)
        count += cv::countNonZero(channels[0]);
    return count;
}


int main()
{
    cv::Mat img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH)+"/galaxy.jpg",
        cv::IMREAD_UNCHANGED
    );
    
    cv::Mat marker;
    cv::morphologyEx(
        img, marker, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(37, 37))
    );
    aia::imshow("Marker", marker);

    cv::Mat marker_cur = marker;
	cv::Mat marker_prev;
	cv::Mat mask = img;
    int iteration = 0;
	do 
	{
		marker_prev = marker_cur.clone();
		cv::dilate(
            marker_cur, marker_cur,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3))
        );
		marker_cur = cv::min(marker_cur, mask);

		// aia::imshow("Reconstruction in progress", marker_cur, false, 0.5);
		// cv::waitKey(100);
        printf("iteration = %d\n", ++iteration);
        
	} while (myCountNonZero(marker_cur - marker_prev));
    
    aia::imshow("Reconstructed image", marker_cur, true, 0.7);
    aia::imshow("Stars image", mask - marker_cur, 0.7);

    return EXIT_SUCCESS;
}
