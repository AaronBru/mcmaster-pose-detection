#include "aruco.hpp"
#include "kabsch.hpp"
#include "detectPoseRealsense.hpp"

#include <opencv2/opencv.hpp>
#include <librealsense2/rsutil.h>

using namespace cv;

void DetectPoseRealsense::processFrame(Mat& origFrame, Mat& depthFrame)
{
//    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);
    //Ptr<aruco::DetectorParameters> arucoParam = aruco::DetectorParameters::create();

    std::vector<std::vector<Point2f>> allMarkers;
    std::vector<std::vector<Point2f>> rejected;
    std::vector<int> markerIds;

    aruco::detectMarkers(origFrame, dictionary, allMarkers, markerIds, arucoParam, rejected);

    int markerId;
    int i = 0;
    foundIds.clear();
    deprojectedPoints.clear();
    for (auto marker = allMarkers.begin(); marker != allMarkers.end(); marker++) {
        markerId = markerIds[i];

        bool validId = false;
        for (unsigned int j = 0; j < validIds.size(); j++) {
            if (validIds[j] == markerId) {
                validId = true;
            }
        }

        i++;
        if (!validId) {
            continue;
        }
    
        float cornerPixel[2];
        float cornerDepth;
        float cornerCoord[3];

        bool validDepth = true;
        int j = 0;

        for (auto markerCorner = marker->begin(); markerCorner != marker->end(); markerCorner++) {
            cornerDepth = depthScale * depthFrame.at<ushort>((int)markerCorner->y, (int)markerCorner->x);
            if (cornerDepth < 0.05) {
                validDepth = false;
                for (int k = 0; k < j; k++) {
                    deprojectedPoints.pop_back();
                }
                break;
            }
            cornerPixel[0] = markerCorner->x;
            cornerPixel[1] = markerCorner->y;

            rs2_deproject_pixel_to_point(cornerCoord, intrinsics, cornerPixel, cornerDepth);
            deprojectedPoints.push_back(Point3f(cornerCoord[0], cornerCoord[1],  cornerCoord[2]));
            j++;
        }
        if (!validDepth) {
            continue;
        }

        foundIds.push_back(markerId);
    }
}


bool DetectPoseRealsense::getPose(std::map<int, Mat> &objectMarkers, Mat &rot, std::array<float, 3> &translation)
{
    if (!foundIds.size()) {
        return false;
    }

    std::vector<int> objectIds;
    for (auto &id : foundIds) {
        if (objectMarkers.count(id) > 0) {
            objectIds.push_back(id);
        }
    }

    if (!objectIds.size()) {
        return false;
    }

    Mat markerCorners = Mat(deprojectedPoints.size(), 3, CV_32FC1, deprojectedPoints.data());

    Mat refCorners;
    if (objectIds.size() == 1) {
        refCorners = objectMarkers[objectIds[0]];
    }
    else if (objectIds.size() > 1) {
        vconcat(objectMarkers[objectIds[0]], objectMarkers[objectIds[1]], refCorners);
        for (size_t i = 2; i < objectIds.size(); i++) {
            vconcat(refCorners, objectMarkers[objectIds[i]], refCorners);
        }
    }
    computeRotAndTrans(markerCorners, refCorners, translation, rot);
    return true;
}


static void addVector(float (*vecA)[3], float (*vecB)[3], float (*vecOut)[3])
{
    for (size_t i = 0; i < 3; i++) {
        (*vecOut)[i] = (*vecA)[i] + (*vecB)[i];
    }
}


static void scaleVector(float (*vec)[3], float scaleFactor)
{

    (*vec)[0] *= scaleFactor;
    (*vec)[1] *= scaleFactor;
    (*vec)[2] *= scaleFactor;
}


void drawMarker(Mat& rot, std::array<float, 3> translation, const struct rs2_intrinsics *intrinsics, Mat &origFrame)
{
    float origPixel[2];
    float xPixel[2];
    float yPixel[2];
    float zPixel[2];

    float xPoint[3] = {rot.at<float>(0, 0), rot.at<float>(1, 0), rot.at<float>(2, 0)};
    float yPoint[3] = {rot.at<float>(0, 1), rot.at<float>(1, 1), rot.at<float>(2, 1)};
    float zPoint[3] = {rot.at<float>(0, 2), rot.at<float>(1, 2), rot.at<float>(2, 2)};
    float translationArr[3] = {translation[0], translation[1], translation[2]};

    scaleVector(&xPoint, 0.1);
    scaleVector(&yPoint, 0.1);
    scaleVector(&zPoint, 0.1);

    addVector(&xPoint, &translationArr, &xPoint);
    addVector(&yPoint, &translationArr, &yPoint);
    addVector(&zPoint, &translationArr, &zPoint);

    rs2_project_point_to_pixel(origPixel, intrinsics, translationArr);
    rs2_project_point_to_pixel(xPixel, intrinsics, xPoint);
    rs2_project_point_to_pixel(yPixel, intrinsics, yPoint);
    rs2_project_point_to_pixel(zPixel, intrinsics, zPoint);


    Point origin = Point(origPixel[0], origPixel[1]);
    Point xAxis  = (Point(xPixel[0],    xPixel[1]) - origin) + origin;
    Point yAxis  = (Point(yPixel[0],    yPixel[1]) - origin) + origin;
    Point zAxis  = (Point(zPixel[0],    zPixel[1]) - origin) + origin;
     
    line(origFrame, origin, xAxis, CV_RGB(200, 50, 50), 2);
    line(origFrame, origin, yAxis, CV_RGB(50, 200, 50), 2);
    line(origFrame, origin, zAxis, CV_RGB(50, 50, 200), 2);
}
