#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "icp.hpp"

using namespace cv;

float getDepthScale(rs2::device dev);

struct rsConfig {
    rs2::pipeline         rsPipe;
    rs2::pipeline_profile rsProfile;
    rs2_stream            rsAlignTo;

    int depthFps;
    int irFps;
    int colorFps;

    std::array<int, 2>  depthRes;
    std::array<int, 2>  irRes;
    std::array<int, 2>  colorRes;

    float depthScale;
};


static void initRsCam(struct rsConfig &rsCfg)
{
    rsCfg.depthRes = {640, 480};
    rsCfg.irRes    = {640, 480};
    rsCfg.colorRes = {1920, 1080};
    rsCfg.colorFps = 30;
    rsCfg.irFps    = 60;
    rsCfg.depthFps = 60;

    //Enable streams
    rs2::config config;
    config.enable_stream(RS2_STREAM_INFRARED, rsCfg.irRes[0], rsCfg.irRes[1], RS2_FORMAT_Y8, rsCfg.irFps);
    config.enable_stream(RS2_STREAM_DEPTH, rsCfg.depthRes[0], rsCfg.depthRes[1], RS2_FORMAT_Z16,  rsCfg.depthFps);

    //Begin rs2 pipeline
    rs2::pipeline pipe;
    rsCfg.rsPipe = pipe;
    rsCfg.rsProfile = rsCfg.rsPipe.start(config);

    rsCfg.depthScale = getDepthScale(rsCfg.rsProfile.get_device());
    //Initialize alignment
    rsCfg.rsAlignTo = RS2_STREAM_INFRARED;
}


static bool displayUi(int xTarget, int yTarget, Mat& image, std::array<float, 3> vector)
{

    std::string displacement = std::to_string(vector[0]) + ", " + std::to_string(vector[1]) + ", " + std::to_string(vector[2]);
    putText(image, displacement, Point(50, 50),  FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 0), 0);

    std::string pixel = std::to_string(xTarget) + ", " + std::to_string(yTarget);
    putText(image, pixel,        Point(50, 100), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 0), 0);

    namedWindow("Show Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    char button;
    button = waitKey(1);

    if (button == 27) {
        destroyAllWindows();
        return true;
    }

    return false;
}


void normVector(float (*vector)[3])
{
    float mag = std::sqrt((*vector)[0] * (*vector)[0] + (*vector)[1] * (*vector)[1] + (*vector)[2] * (*vector)[2]);
    (*vector)[0] = (*vector[0]) / mag;
    (*vector)[1] = (*vector[1]) / mag;
    (*vector)[2] = (*vector[2]) / mag;
}


static float originalPoints[3][3] = {{-0.13/3,        -0.1/3,      0},
                                     {0.1 - 0.13/3,   -0.1/3,      0},
                                     {0.03 - 0.13/3,   0.1-0.1/3,  0}};

std::array<int, 2> processFrame(Mat& origFrame, Mat& depthFrame, const struct rs2_intrinsics* intrinsics, float depthScale, Mat& outFrame)
{
    Mat freshClone = origFrame.clone();

    GaussianBlur(origFrame, origFrame, cv::Size(5, 5), 3.0, 3.0);

    threshold(origFrame, origFrame, 100, 255, CV_THRESH_TOZERO);

    std::vector<std::vector<Point>> contours;

    int threshold = 50;
    Canny(origFrame, origFrame, threshold, 3 * threshold);

    std::vector<Vec4i> hierarchy;
    findContours(origFrame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<Point>> markers;
    std::vector<Rect> markerBoxes;
    std::vector<int> markersArea;

    float ratio;
    int area;
    for (size_t i = 0; i < contours.size(); i++) {
        
        Rect boundBox;
        if (hierarchy[i][2] < 0 && hierarchy[i][3] < 0
         || hierarchy[i][2] > 0) {

            boundBox = boundingRect(contours[i]);
            ratio = ((float)boundBox.width / (float)boundBox.height);
            area = boundBox.area();

            if (ratio > 0.70 && ratio < 1.30 && area > 20) {
                markers.push_back(contours[i]);
                markerBoxes.push_back(boundBox);
                markersArea.push_back(area);
            }
        }
    }

    cvtColor(freshClone, outFrame, CV_GRAY2RGB);
    float points[3][3];

    std::vector<int> largestMarkersArea = {-1, -1, -1};
    std::vector<std::vector<Point>> largestMarkers;
    bool allMarkersFound = false;
    if (markers.size() >= 3) {
        allMarkersFound = true;

        largestMarkers.push_back(markers[0]);
        largestMarkers.push_back(markers[1]);
        largestMarkers.push_back(markers[2]);

        largestMarkersArea.push_back(markersArea[0]);
        largestMarkersArea.push_back(markersArea[1]);
        largestMarkersArea.push_back(markersArea[2]);

        for (size_t i = 3; i < markers.size(); i++) {
            if (markersArea[i] > largestMarkersArea[0]) {
                largestMarkers[2] = largestMarkers[1];
                largestMarkers[1] = largestMarkers[0];
                largestMarkers[0] = markers[i];

                largestMarkersArea[2] = largestMarkersArea[1];
                largestMarkersArea[1] = largestMarkersArea[0];
                largestMarkersArea[0] = markersArea[i];
            }
            else if (markersArea[i] > largestMarkersArea[1]) {
                largestMarkers[2] = largestMarkers[1];
                largestMarkers[1] = markers[i];

                largestMarkersArea[2] = largestMarkersArea[1];
                largestMarkersArea[1] = markersArea[i];
            }
            else if (markersArea[i] > largestMarkersArea[2]) {
                largestMarkers[2] = markers[i];
                largestMarkersArea[2] = markersArea[i];
            }
        }
    }

    float distPoint[2] = {0, 0};
    float outPoint[3] = {0, 0, 0};
    if (allMarkersFound) {
        bool allDepthsValid = true;
        for (size_t i = 0; i < largestMarkers.size(); i++) {
            Rect bbox = boundingRect(largestMarkers[i]);
            distPoint[0] = bbox.x + bbox.width  / 2;
            distPoint[1] = bbox.y + bbox.height / 2;
            float depth = depthScale * depthFrame.at<ushort>((int)distPoint[1], (int)distPoint[0]);

            if (depth == 0) {
                allDepthsValid = false;
                break;
            }

            rs2_deproject_pixel_to_point(outPoint, intrinsics, distPoint, depth);
            
            for (size_t j = 0; j < 3; j++) {
                points[i][j] = outPoint[j];
            }
        }
        
        Scalar colorList[3] = {CV_RGB(200, 50, 50), CV_RGB(50, 200, 50), CV_RGB(50, 50, 200)};
        for (size_t i = 0; i < 3; i++) {
            drawContours(outFrame, largestMarkers, i, colorList[i], 3);
            rectangle(outFrame, boundingRect(largestMarkers[i]), colorList[i], 3);
        }
        
        if (allDepthsValid) {

            float centroid[3] = {0, 0, 0};
            for (size_t i = 0; i < 3; i++) {
                centroid[0] += points[i][0];
                centroid[1] += points[i][1];
                centroid[2] += points[i][2];
            }

            centroid[0] /= 3.0;
            centroid[1] /= 3.0;
            centroid[2] /= 3.0;

            for (size_t i = 0; i < 3; i++) {
                points[i][0] -= centroid[0];
                points[i][1] -= centroid[1];
                points[i][2] -= centroid[2];
            }

            float dx1 = points[0][0] - points[1][0];
            float dx2 = points[0][0] - points[2][0];
            float dx3 = points[1][0] - points[2][0];

            float dy1 = points[0][1] - points[1][1];
            float dy2 = points[0][1] - points[2][1];
            float dy3 = points[1][1] - points[2][1];

            float dz1 = points[0][2] - points[1][2];
            float dz2 = points[0][2] - points[2][2];
            float dz3 = points[1][2] - points[2][2];

            float d1 = std::sqrt(dx1 * dx1  + dy1 * dy1 + dz1 * dz1);
            float d2 = std::sqrt(dx2 * dx2  + dy2 * dy2 + dz2 * dz2);
            float d3 = std::sqrt(dx3 * dx3  + dy3 * dy3 + dz3 * dz3);

            int originMarker, parallel, perpendicular;
            if (d1 > d2 && d1 > d3) {
                originMarker = 2;

                if (d2 > d3) {
                    parallel      = 1;
                    perpendicular = 0;
                }
                else {
                    parallel      = 0;
                    perpendicular = 1;
                }

            }
            else if (d2 > d3 && d2 > d1) {
                originMarker = 1;

                if (d3 > d1) {
                    parallel      = 0;
                    perpendicular = 2;
                }
                else {
                    parallel      = 2;
                    perpendicular = 0;
                }

            }
            else if (d3 > d1 && d3 > d2) {
                originMarker = 0;

                if (d1 > d2) {
                    parallel      = 2;
                    perpendicular = 1;
                }
                else {
                    parallel      = 1;
                    perpendicular = 2;
                }
            }

            float sortedPoints[3][3];
            for (size_t i = 0; i < 3; i++) {
                sortedPoints[0][i] = points[originMarker][i]; 
                sortedPoints[1][i] = points[parallel][i]; 
                sortedPoints[2][i] = points[perpendicular][i]; 
            }

            Mat refPoints(Size(3, 3), CV_32FC1, originalPoints);
            transpose(refPoints, refPoints);

            Mat newPoints(Size(3, 3), CV_32FC1, sortedPoints);
            Mat crossCov(Size(3, 3), CV_32FC1);
            crossCov = refPoints * newPoints;

            Mat Ut  = Mat(Size(3, 3), CV_32FC1);
            Mat S  = Mat(Size(3, 3), CV_32FC1);
            Mat V = Mat(Size(3, 3), CV_32FC1);

            
            
            SVD::compute(crossCov, S, Ut, V);
            transpose(Ut, Ut);
            transpose(V, V);

            Mat rot(Size(3, 3), CV_32FC1);
            float signCorrection = (determinant(V*Ut) > 0) ? 1 : -1;
            float correctFloat[3][3] = {{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, signCorrection}};
            Mat correct(Size(3, 3), CV_32FC1, correctFloat);
            rot = V * correct * Ut;

            std::cout << S.at<float>(0,0) << " " << S.at<float>(1,1) << " " << S.at<float>(2,2) << "\n";

            float origPixel[2];
            float xPixel[2];
            float yPixel[2];
            float zPixel[2];

            float xPoint[3] = {rot.at<float>(0, 0)/3, rot.at<float>(1, 0)/3, rot.at<float>(2, 0)/3};
            float yPoint[3] = {rot.at<float>(0, 1)/3, rot.at<float>(1, 1)/3, rot.at<float>(2, 1)/3};
            float zPoint[3] = {rot.at<float>(0, 2)/3, rot.at<float>(1, 2)/3, rot.at<float>(2, 2)/3};

            rs2_project_point_to_pixel(origPixel, intrinsics, centroid);
            rs2_project_point_to_pixel(xPixel, intrinsics, xPoint);
            rs2_project_point_to_pixel(yPixel, intrinsics, yPoint);
            rs2_project_point_to_pixel(zPixel, intrinsics, zPoint);

            Point origin = Point(origPixel[0], origPixel[1]);
            Point xAxis  = (Point(xPixel[0],    xPixel[1]) - origin)  / 10 + origin;
            Point yAxis  = (Point(yPixel[0],    yPixel[1]) - origin)  / 10 + origin;
            Point zAxis  = (Point(zPixel[0],    zPixel[1]) - origin)  / 10 + origin;
             
            line(outFrame, origin, xAxis, CV_RGB(200, 50, 50), 2);
            line(outFrame, origin, yAxis, CV_RGB(50, 200, 50), 2);
            line(outFrame, origin, zAxis, CV_RGB(50, 50, 200), 2);
        }
    }

    return std::array<int, 2> {0, 0};
}


float getDepthScale(rs2::device dev)
{
    for (rs2::sensor& sensor : dev.query_sensors()) {
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>()) {
            return dpt.get_depth_scale();
        }
    }
    return 1.0;
}


int main() 
{
    struct rsConfig rsCfg;
    initRsCam(rsCfg);

    auto stream = rsCfg.rsProfile.get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
    const struct rs2_intrinsics intrinsics = stream.get_intrinsics();

    float vPixel[2];
    float vPoint[3] = {0,0,0};

    rs2::align align(rsCfg.rsAlignTo);

    while(true) {

        rs2::frameset frames = rsCfg.rsPipe.wait_for_frames();
        auto processed = align.process(frames);

        rs2::video_frame other_frame = processed.first_or_default(rsCfg.rsAlignTo);
        rs2::depth_frame depth       = processed.get_depth_frame();

        if (!other_frame || !depth) {
            continue;
        }

        Mat origFrame(Size(rsCfg.irRes[0], rsCfg.irRes[1]), CV_8UC1, (void*)other_frame.get_data(), Mat::AUTO_STEP);
        Mat depthFrame(Size(rsCfg.depthRes[0], rsCfg.depthRes[1]), CV_16U, (void*)depth.get_data(), Mat::AUTO_STEP);
        Mat outFrame(Size(rsCfg.irRes[0], rsCfg.irRes[1]), CV_8UC3);
        std::array<int, 2> trackPos = processFrame(origFrame, depthFrame, &intrinsics, rsCfg.depthScale, outFrame);

        std::array<float, 3> vec = {0, 0, 0};
        if (displayUi(0, 0, outFrame,  vec)) {
            break;
        }
    }
    return 0;
}
