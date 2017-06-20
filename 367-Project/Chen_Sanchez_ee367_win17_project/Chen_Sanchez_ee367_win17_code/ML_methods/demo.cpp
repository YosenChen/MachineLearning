#include "opencv2/ml.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "vector"
#include <iostream>
#include <cstdio>       /* printf */
#include <assert.h>     /* assert */
#include <cmath>
#include <algorithm>    /* random_shuffle */

using namespace std;
using namespace cv;

int main() {

    auto img = imread("..//ex_image.PNG");
    if (img.empty()) printf("fail to load.\n");

    Mat roi = img(Rect(0, 0, img.cols/3, img.rows/3));
    imwrite("ex_image_roi.png", roi);
    Mat bgr[3];
    split(roi, bgr);
    imwrite("ex_image_roi_B.png", bgr[0]);
    imwrite("ex_image_roi_G.png", bgr[1]);
    imwrite("ex_image_roi_R.png", bgr[2]);
    
    auto det = imread("detail_map_i22_k15.png");
    if (det.empty()) printf("fail to load (2).\n");

    split(det, bgr);
    imwrite("det_B.png", bgr[0]);

    auto orig = imread("../ex_image_color.PNG");
    if (orig.empty()) printf("fail to load (3).\n");

    roi = orig(Rect(0, 0, orig.cols/3, orig.rows/3));
    imwrite("ex_image_color_roi.png", roi);
    split(roi, bgr);
    imwrite("ex_image_color_roi_B.png", bgr[0]);
    imwrite("ex_image_color_roi_G.png", bgr[1]);
    imwrite("ex_image_color_roi_R.png", bgr[2]);



    return 0;
}



