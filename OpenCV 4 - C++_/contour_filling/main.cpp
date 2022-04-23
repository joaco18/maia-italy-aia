#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
    // Define the rectangles
    cv::Mat img(800, 800, CV_8U, cv::Scalar(0));
    cv::rectangle(img, cv::Rect(50, 50, 100, 150), cv::Scalar(255), 2);
    cv::rectangle(img, cv::Rect(400, 400, 200, 250), cv::Scalar(255), 2);
    aia::imshow("Image", img);
    
    // Filling
    cv::Mat marker = img.clone();
    marker.setTo(cv::Scalar(0));
    cv::rectangle(
        marker, cv::Rect(0, 0, marker.cols - 1, marker.rows -1), cv::Scalar(255)
    );
    cv::Mat mask = 255 - img;
    ucas::imshow("Hole filling (mask)", mask);
    cv::Mat marker_prev;
    do
    {
        marker_prev = marker.clone();
        cv::Mat SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(marker, marker, SE);
        marker = marker & mask;

        cv::imshow("Hole filling (in progress)", marker);
        if (cv::waitKey(50) >= 0)
            cv::destroyWindow("Hole filling (in progress)");
    } while (cv::countNonZero(marker - marker_prev));

    ucas::imshow("Hole filling (result)", 255 -marker);
    return EXIT_SUCCESS;
}