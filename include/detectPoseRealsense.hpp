#ifndef _DETECTPOSEREALSENSE_HPP
#define _DETECTPOSEREALSENSE_HPP

#include "aruco.hpp"

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>

#include <map>
#include <vector>
#include <array>

class DetectPoseRealsense {

    public:
        const struct rs2_intrinsics *intrinsics;
        float depthScale;
        DetectPoseRealsense(const struct rs2_intrinsics *intrinsics, float depthScale, std::vector<int> validIds):
        intrinsics(intrinsics), depthScale(depthScale), validIds(validIds), deprojectedPoints()
        {
            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
            arucoParam = cv::aruco::DetectorParameters::create();

#if 0
            for (auto id : validIds) {
                std::vector<cv::Point3f> empty;
                deprojectedPoints[id] = empty;
            }
#endif
        }

        void processFrame(cv::Mat& origFrame, cv::Mat& depthFrame);
        bool getPose(std::map<int, cv::Mat> &objectMarkers, cv::Mat &rot, std::array<float, 3> &translation);

    private:
        std::vector<int> validIds;
	std::map<int, std::vector<cv::Point3f>> deprojectedPoints;
        //std::vector<cv::Point3f> deprojectedPoints;
        std::vector<int> foundIds;
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        cv::Ptr<cv::aruco::DetectorParameters> arucoParam;
};

void drawMarker(cv::Mat& rot, std::array<float, 3> translation, const struct rs2_intrinsics *intrinsics, cv::Mat &origFrame);
#endif

