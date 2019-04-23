#include "aruco.hpp"
#include "kabsch.hpp"
#include "detectPoseRealsense.hpp"

#include <opencv2/opencv.hpp>
#include <librealsense2/rsutil.h>

using namespace cv;

/* processFrame
Pass the colorframe and depth frame from librealsense to this function here to detect any ids
that the class is set to detect (via validIds in the constructor) */
void DetectPoseRealsense::processFrame(Mat& colorFrame, Mat& depthFrame)
{
    std::vector<std::vector<Point2f>> allMarkers;
    std::vector<std::vector<Point2f>> rejected;
    std::vector<int> markerIds;

    aruco::detectMarkers(colorFrame, dictionary, allMarkers, markerIds, arucoParam, rejected);

    int markerId;
    int i = 0;
    foundIds.clear();
    for (auto marker = allMarkers.begin(); marker != allMarkers.end(); marker++) {
        markerId = markerIds[i];

        bool validId = false;
        /* Filter out unused ids */
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

        int j = 0;

        bool validDepth = true;
        deprojectedPoints[markerId].clear();
        for (auto markerCorner = marker->begin(); markerCorner != marker->end(); markerCorner++) {
            cornerDepth = depthScale * depthFrame.at<ushort>((int)markerCorner->y, (int)markerCorner->x);
            /* Check for a valid depth. If not valid, remove all points related to the marker */
            if (cornerDepth < 0.05) {
                validDepth = false;
                deprojectedPoints[markerId].clear();
                break;
            }
            cornerPixel[0] = markerCorner->x;
            cornerPixel[1] = markerCorner->y;

            rs2_deproject_pixel_to_point(cornerCoord, intrinsics, cornerPixel, cornerDepth);
            deprojectedPoints[markerId].push_back(Point3f(cornerCoord[0], cornerCoord[1],  cornerCoord[2]));
            j++;
        }
        if (!validDepth) {
            continue;
        }
        foundIds.push_back(markerId);
    }
}


/* getPose
After calling processFrame, call this function with a map of associated Aruco marker ids and corner positions to get
the rotation and translation of the object */
bool DetectPoseRealsense::getPose(std::map<int, Mat> &objectMarkers, Mat &rot, std::array<float, 3> &translation)
{
    if (!foundIds.size()) {
        return false;
    }

    /* Determine which of the detected ids are also object ids */
    std::vector<int> objectIds;
    std::vector<Point3f> cornerPoints;
    for (auto &id : foundIds) {
        if (objectMarkers.count(id) > 0) {
            objectIds.push_back(id);
            cornerPoints.insert(std::end(cornerPoints), std::begin(deprojectedPoints[id]),
                                                        std::end(deprojectedPoints[id]));
#if 0
            for (auto elem : deprojectedPoints[id]) {
                cornerPoints.push_back(elem);
            }
#endif
        }
    }

    if (!objectIds.size()) {
        return false;
    }

    Mat markerCorners = Mat(cornerPoints.size(), 3, CV_32FC1, cornerPoints.data());

    Mat refCorners;

    /* Build a matrix of corresponding reference and observed points */
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


/* Helper function for use in draw markers */
static void addVector(float (*vecA)[3], float (*vecB)[3], float (*vecOut)[3])
{
    for (size_t i = 0; i < 3; i++) {
        (*vecOut)[i] = (*vecA)[i] + (*vecB)[i];
    }
}


/* Helper function for use in draw markers */
static void scaleVector(float (*vec)[3], float scaleFactor)
{

    (*vec)[0] *= scaleFactor;
    (*vec)[1] *= scaleFactor;
    (*vec)[2] *= scaleFactor;
}


/* Helper function to draw markers */
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
