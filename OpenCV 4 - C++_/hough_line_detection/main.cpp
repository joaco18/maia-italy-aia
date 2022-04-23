#include "aiaConfig.h"
#include "ucasConfig.h"

// GOAL: show how the Hough transform can be used to detect lines
//       we also build a minimal Graphical User Interface (GUI) so that we can
//       interactively select the parameters and see the result on the image

// NOTE: the Hough transform works on binary images only. To generate a binary
//       image suited for Hough, we have to detect edges first. Here, this is 
//       done with image derivatives, but in general it can also be done with
//       other methods (e.g. Canny, Marr-Hildtreth, custom criteria, etc.)

// since we work with a GUI, we need parameters (and the images) to be stored
// in global variables
namespace aia
{
    cv::Mat img;
    cv::Mat imgEdges;

    int stdevX10;
    int threshold;
    int alpha0;
    int alpha1;

    int drho;
    int dtheta;
    int accum;
    int n;

    void edgeDetectionGrad(int, void*)
    {
        if (stdevX10 > 0)
            cv::GaussianBlur(
                img, imgEdges, cv::Size(0, 0), (stdevX10/10.0), (stdevX10/10.0)
            );
        else
            imgEdges = img.clone();
        cv::Mat img_dx, img_dy;
        cv::Sobel(imgEdges, img_dx, CV_32F, 1, 0);
        cv::Sobel(imgEdges, img_dy, CV_32F, 0, 1);

        cv::Mat mag, angle;
        cv::cartToPolar(img_dx, img_dy, mag, angle, true);

        for (int y = 0; y < imgEdges.rows; y++)
        {
            aia::uint8* imgEdgesYthRow = imgEdges.ptr <aia::uint8>(y);
            float* magYthRow = mag.ptr<float>(y);
            float* angleYthRow = angle.ptr<float>(y);
            for (int x = 0; x < imgEdges.cols; x++)
            {
                if (magYthRow[x] > threshold && (
                    angleYthRow[x] >= alpha0 || angleYthRow[x] <= alpha1)
                )
                    imgEdgesYthRow[x] = 255;
                else
                    imgEdgesYthRow[x] = 0;
            }
        }
        cv::imshow("Edge detection (gradient)", imgEdges);
    }

    void Hough(int, void*)
    {
        // If invalid parameters don't do noth
        if (drho <= 0)
            return;
        if (dtheta <= 0)
            return;
        if (accum <= 0)
            return;
        
        // Hough returns a vector of lines represented by (rho,theta) pairs
		// the vector is automatically sorted by decreasing accumulation scores
		// this means the first 'n' lines are the most voted
        std::vector<cv::Vec2f> lines;
        cv::HoughLines(imgEdges, lines, drho, dtheta/180.0, accum);
        
        // draw first n lines
        cv::Mat img_copy = img.clone();
        for (int k = 0; k < std::min(size_t(n), lines.size()); k++)
        {   // ~vertical line
            float rho = lines[k][0];
            float theta = lines[k][1];
            if (theta < aia::PI / 4. || theta >3. * aia::PI / 4.)
            {
                // point of intersection of the line with the first row
                cv::Point pt1(rho / cos(theta), 0);
                // point of intersection of the line with last row
                cv::Point pt2(
                    (rho - img_copy.rows * sin(theta)) / cos(theta), img_copy.rows
                );
                // draw a white line
                cv::line(img_copy, pt1, pt2, cv::Scalar(0, 0, 255), 1);
            }
            else
            { // ~horizontal line
                //point of intersection of the line with first column
                cv::Point pt1(0, rho / sin(theta));
                // point of intersection of the line with last column
                cv::Point pt2(
                    img_copy.cols, (rho -img_copy.cols * cos(theta)) / sin(theta)
                );
                // draw a white line
                cv::line(
                    img_copy, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA
                );
            }
        }
        cv::imshow("Line detection (Hough)", img_copy);
    }
}

int main()
{
    try
    {
        aia::img = cv::imread(
            std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg", cv::IMREAD_GRAYSCALE)
        ;
        if (!aia::img.data)
            throw aia::error("cannot open image");

        // default paramenters
        aia::stdevX10 = 10;
        aia::threshold = 60;
        aia::alpha0 = 0;
        aia::alpha1 = 360;
        aia::drho = 1;
        aia::dtheta = 1;
        aia::accum = 1;
        aia::n = 10;

        std::string grad_win_name = "Edge detection (gradient)";
        cv::namedWindow(grad_win_name);
        cv::createTrackbar(
            "stdev(x10)", grad_win_name, &aia::stdevX10,
            100, aia::edgeDetectionGrad
        );
        cv::createTrackbar(
            "threshold", grad_win_name, &aia::threshold,
            100, aia::edgeDetectionGrad
        );
        cv::createTrackbar(
            "alpha0", grad_win_name, &aia::alpha0,
            360, aia::edgeDetectionGrad
        );
        cv::createTrackbar(
            "alpha1", grad_win_name, &aia::alpha1,
            360, aia::edgeDetectionGrad
        );

        std::string hough_win_name = "Edge detection (Hough)";
        cv::namedWindow(hough_win_name);
        cv::createTrackbar("drho", hough_win_name, &aia::drho, 100, aia::Hough);
        cv::createTrackbar(
            "dtheta", hough_win_name, &aia::dtheta, 100, aia::Hough
        );
        cv::createTrackbar("accum", hough_win_name, &aia::accum, 360, aia::Hough);
        cv::createTrackbar("n", hough_win_name, &aia::n, 360, aia::Hough);

        aia::edgeDetectionGrad(1, 0);
        aia::Hough(1, 0);
        cv::waitKey(0);

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