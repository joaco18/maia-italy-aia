#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
    std::string win_name = "Object picking";
    cv::Mat img;
    std::vector <std::vector <cv::Point>> objects_filtered;
    void objectPickingCallback(int event, int x, int y, int, void*)
    {
        cv::Mat img_copy = img.clone();
        if (event ==cv::EVENT_LBUTTONDOWN)
        {
            for (int i = 0; i < objects_filtered.size(); i++)
                if (cv::pointPolygonTest(
                        objects_filtered[i], cv::Point(x,y), false
                    ) > 0)
                {
                    cv::drawContours(
                        img_copy, objects_filtered, i, cv::Scalar(255, 255, 255),
                        cv::FILLED, cv::LINE_AA
                    );
                }
                cv:imshow(win_name, img_copy);
        }
    }
}

int main()
{
    img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH)+"/tools.png",
        cv::IMREAD_GRAYSCALE
    );
    
    int T = ucas::getTriangleAutoThreshold(ucas::histogram(img));
    cv::Mat binarized_img;
    cv::threshold(img, binarized_img, T, 255, cv::THRESH_BINARY);

    std::vector <std::vector <cv::Point>> objects;
    cv::findContours(
        binarized_img, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE
    );
    printf("objects (before filtering) = %d\n", objects.size());

    objects_filtered;
    for (int i =0; i < objects.size(); i++)
    {
        double A = cv::contourArea(objects[i]);
        if (A > 100)
            objects_filtered.push_back(objects[i]);
    }
    printf("objects (after filtering) = %d\n", objects_filtered.size());

    cv::namedWindow(win_name);
    cv::setMouseCallback(win_name, objectPickingCallback);
    objectPickingCallback(0, 0, 0, 0, 0);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}   
