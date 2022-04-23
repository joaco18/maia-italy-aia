#include "aiaConfig.h"
#include "ucasConfig.h"


cv::Mat rotate90(cv::Mat img, int step);

int main()
{
    cv::Mat img = cv::imread(
        std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree.tif",
        cv::IMREAD_GRAYSCALE
    );

    // Thinning SEs
    cv::Mat base_north_horizontal_SE = (
        cv::Mat_<int>(3,3)  <<
         -1,  -1,  -1,
          0,   1,   0,
          1,   1,   1);
    cv::Mat base_northeast_diagonal_SE = (
        cv::Mat_<int>(3,3)  <<
          0,  -1,  -1,
          1,   1,  -1,
          1,   1,   0);
    
    std::vector <cv::Mat> thinning_SEs;
    thinning_SEs.push_back(base_north_horizontal_SE);
    thinning_SEs.push_back(rotate90(base_north_horizontal_SE, 1));
    thinning_SEs.push_back(rotate90(base_north_horizontal_SE, 2));
    thinning_SEs.push_back(rotate90(base_north_horizontal_SE, 3));
    thinning_SEs.push_back(base_northeast_diagonal_SE);
    thinning_SEs.push_back(rotate90(base_northeast_diagonal_SE, 1));
    thinning_SEs.push_back(rotate90(base_northeast_diagonal_SE, 2));
    thinning_SEs.push_back(rotate90(base_northeast_diagonal_SE, 3));

    // skeletonization
    cv::Mat current = img.clone();
    cv::Mat previous;
    int k = 0;
    do
    {
        printf("Iteration no. %d\n", ++k);
        previous = current.clone();
        for (int i =0; i < thinning_SEs.size(); i++)
        {
            cv::Mat hm_res;
            cv::morphologyEx(current, hm_res, cv::MORPH_HITMISS, thinning_SEs[i]);
            current -= hm_res;
        }
        cv::imshow("Skeletonization in progress", current);
        cv::waitKey(100);
    } while (cv::countNonZero(previous - current));

    aia::imshow("Skeletonization result", current);
    cv::Mat skeleton = current;
    cv::imwrite(
        std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_skeleton.tif",
        skeleton
    );

    // PRUNING SEs
    cv::Mat base_pruning_SE = (
        cv::Mat_<int>(3, 3)  <<
             0,     0,     0,
            -1,     1,    -1,
            -1,    -1,    -1);

    std::vector <cv::Mat> prunning_SEs;
    prunning_SEs.push_back(base_pruning_SE);
    prunning_SEs.push_back(rotate90(base_pruning_SE, 1));
    prunning_SEs.push_back(rotate90(base_pruning_SE, 2));
    prunning_SEs.push_back(rotate90(base_pruning_SE, 3));

    // pruning
    printf("Prunning");
    int prunning_iters = 5;
    current = skeleton.clone();
    for (int k=0; k < prunning_iters; k++)
    {
        printf("Iteration no. %d\n", k);
        previous = current.clone();
        for (int i = 0; i < prunning_SEs.size(); i++)
        {
            cv::Mat hm_res;
            cv::morphologyEx(current, hm_res, cv::MORPH_HITMISS, prunning_SEs[i]);
            current -= hm_res;
        }
        cv::imshow("Prunning in progress", current);
        cv::waitKey(100);
    }
    aia::imshow("Prunning result", current);
    cv::Mat skeleton_pruned = current;
    cv::imwrite(
        std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_pruning.tif",
        skeleton_pruned
    );

    // define base junctions SEs
    cv::Mat base_junctions1_SE = (
        cv::Mat_<int>(3, 3)  <<
             0,    -1,     0,
             1,     1,     1,
            -1,     1,     0);
    
    cv::Mat base_junctions2_SE = (
        cv::Mat_<int>(3, 3)  <<
             0,    -1,     0,
             1,     1,     1,
             0,     1,    -1);

    std::vector <cv::Mat> junctions_SEs;
    junctions_SEs.push_back(base_junctions1_SE);
    junctions_SEs.push_back(rotate90(base_junctions1_SE, 1));
    junctions_SEs.push_back(rotate90(base_junctions1_SE, 2));
    junctions_SEs.push_back(rotate90(base_junctions1_SE, 3));
    junctions_SEs.push_back(base_junctions2_SE);
    junctions_SEs.push_back(rotate90(base_junctions2_SE, 1));
    junctions_SEs.push_back(rotate90(base_junctions2_SE, 2));
    junctions_SEs.push_back(rotate90(base_junctions2_SE, 3));
    
    // T-junctions patter matching
    cv::Mat junctions_image(img.rows, img.cols, CV_8U, cv::Scalar(0));
    for (int i = 0; i < junctions_SEs.size(); i++)
    {
        cv::Mat hm_res;
        cv::morphologyEx(skeleton_pruned, hm_res, cv::MORPH_HITMISS, junctions_SEs[i]);
        junctions_image += hm_res;
    }
    cv::Mat SE = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::dilate(junctions_image, junctions_image, SE);
    aia::imshow("Juncions image", junctions_image);

    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    img.setTo(cv::Scalar(0, 0, 255), junctions_image);
    cv::imwrite(
        std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_junctions.tif", img
    );

    return EXIT_SUCCESS;
}


cv::Mat rotate90(cv::Mat img, int step)
{
    cv::Mat img_rot;

    //adjust step in case it is negative
    if (step < 0)
        step = - step;
    //adjust step in case it exceeds 4
    step = step % 4;
    
    if (step == 0)
        img_rot = img;
    else if (step == 1)
    {
        cv::transpose(img, img_rot);
        cv::flip(img_rot, img_rot, 1);
    }
    else if (step == 2)
    {
        cv::transpose(img, img_rot);
        cv::flip(img_rot, img_rot, -1);
    }
    else if (step == 3)
    {
        cv::transpose(img, img_rot);
        cv::flip(img_rot, img_rot, 0);
    }
    return img_rot;
}