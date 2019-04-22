#ifndef _KABSCH_HPP
#define _KABSCH_HPP

#include <opencv2/opencv.hpp>

void computeRotAndTrans(cv::Mat &points, cv::Mat &origPoints, std::array<float, 3> &translation, cv::Mat& rot);

#endif
