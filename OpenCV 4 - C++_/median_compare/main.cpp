#include "aiaConfig.h"
#include "ucasConfig.h"

// Compute the median filter and get the time profile

int main()
{
    ////////// PARAMETERS /////////
    int k = 7;
    int h = k / 2;
    int m = (k * k) / 2;  // median index

    cv::Mat img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", cv::IMREAD_GRAYSCALE
    );
    cv::Mat out(img.rows, img.cols, CV_8U, cv::Scalar(0));

    ucas::Timer timer;
    
    // filter buffer preallocation
    std::vector<unsigned char> filter_buff(k*k);
    for (int y = h; y < img.rows - h; y++)
    {
        unsigned char* out_yRow = out.ptr<unsigned char>(y);
        for (int x = h; x < img.cols - h; x++)
        {
            cv::Mat filter(img, cv::Rect(x-h, y-h, k, k));
            int i = 0;
            for (int yy = 0; yy < filter.rows; yy++)
            {
                unsigned char* filter_yRow = filter.ptr<unsigned char>(yy);
                for (int xx = 0; xx < filter.cols; xx++)
                {
                    filter_buff[i++] = filter_yRow[xx];
                }
            }
            std::nth_element(
                filter_buff.begin(), filter_buff.begin() + m, filter_buff.end()
            );
            out_yRow[x] = filter_buff[m];
        }
    }
	printf(
        "Elapsed time (custom median filter) = %.3f seconds\n",
        timer.elapsed<float>()
    );

    timer.restart();
    cv::Mat result;
    cv::medianBlur(img, result, k);
    printf(
        "Elapsed time (OpenCV median filter) = %.3f seconds\n",
        timer.elapsed<float>()
    );
    return EXIT_SUCCESS;
}