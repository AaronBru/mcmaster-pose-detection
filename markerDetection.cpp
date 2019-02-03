#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cmath>

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
    rsCfg.colorRes = {640, 480};
    rsCfg.colorFps = 30;
    rsCfg.irFps    = 60;
    rsCfg.depthFps = 30;

    //Enable streams
    rs2::config config;
    config.enable_stream(RS2_STREAM_COLOR, rsCfg.colorRes[0], rsCfg.colorRes[1], RS2_FORMAT_BGR8, rsCfg.colorFps);
    config.enable_stream(RS2_STREAM_DEPTH, rsCfg.depthRes[0], rsCfg.depthRes[1], RS2_FORMAT_Z16,  rsCfg.depthFps);

    //Begin rs2 pipeline
    rs2::pipeline pipe;
    rsCfg.rsPipe = pipe;
    rsCfg.rsProfile = rsCfg.rsPipe.start(config);

    rsCfg.depthScale = getDepthScale(rsCfg.rsProfile.get_device());
    //Initialize alignment
    rsCfg.rsAlignTo = RS2_STREAM_COLOR;
}


static bool displayUi(int xTarget, int yTarget, Mat& image)
{
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


#define POINTS_PER_MARKER 4
static float originalPoints[POINTS_PER_MARKER][3] = {{-0.025,  0.025, 0},
                                                     { 0.025,  0.025, 0},
                                                     { 0.025, -0.025, 0},
                                                     {-0.025, -0.025, 0}};



static void computeRotAndTrans(Mat& points, std::vector<float> &translation, Mat& rot)
{
    int numPoints = points.rows;
    for (int i = 0 ; i < numPoints; i++) {
        translation[0] += points.at<float>(i, 0);
        translation[1] += points.at<float>(i, 1);
        translation[2] += points.at<float>(i, 2);
    }

    translation[0] /= numPoints;
    translation[1] /= numPoints;
    translation[2] /= numPoints;

    for (int i = 0; i < numPoints; i++) {
        points.at<float>(i, 0) -= translation[0];
        points.at<float>(i, 1) -= translation[1];
        points.at<float>(i, 2) -= translation[2];
    }

    Mat refPoints(Size(3, POINTS_PER_MARKER), CV_32FC1, originalPoints);
    transpose(refPoints, refPoints);
    Mat crossCov = refPoints * points;

    Mat Ut;
    Mat S;
    Mat V;

    SVD::compute(crossCov, S, Ut, V);
    transpose(Ut, Ut);
    transpose(V, V);

    float signCorrection = (determinant(V*Ut) > 0) ? 1 : -1;
    float correctFloat[3][3] = {{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, signCorrection}};

    Mat correct(Size(3, 3), CV_32FC1, correctFloat);
    rot = V * correct * Ut;

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


static void processFrame(Mat& origFrame, Mat& depthFrame, const struct rs2_intrinsics* intrinsics, float depthScale)
{
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);
    Ptr<aruco::DetectorParameters> arucoParam = aruco::DetectorParameters::create();

    std::vector<std::vector<Point2f>> allMarkers;
    std::vector<std::vector<Point2f>> rejected;
    std::vector<int> markerIds;

    aruco::detectMarkers(origFrame, dictionary, allMarkers, markerIds, arucoParam, rejected);

    std::vector<std::vector<float>> allTranslations;
    std::vector<Mat> allRotations;
    Mat rot(Size(3,3), CV_32FC1);
    std::vector<float> translation(3);
    bool markerFound = false;

    for (auto marker = allMarkers.begin(); marker != allMarkers.end(); marker++) {
        markerFound = true;
        float cornerPixel[2];
        float cornerDepth;
        float cornerCoord[3];
        std::vector<Point3f> deprojectedPoints;

        for (auto markerCorner = marker->begin(); markerCorner != marker->end(); markerCorner++) {
            cornerDepth = depthScale * depthFrame.at<ushort>((int)markerCorner->y, (int)markerCorner->x);
            cornerPixel[0] = markerCorner->x;
            cornerPixel[1] = markerCorner->y;

            rs2_deproject_pixel_to_point(cornerCoord, intrinsics, cornerPixel, cornerDepth);
            deprojectedPoints.push_back(Point3f(cornerCoord[0], cornerCoord[1],  cornerCoord[2]));
        }
        Mat markerCorners = Mat(deprojectedPoints.size(), 3, CV_32FC1, deprojectedPoints.data());

        Mat rot(Size(3,3), CV_32FC1);
        std::vector<float> translation(3);
        computeRotAndTrans(markerCorners, translation, rot);

        allRotations.push_back(rot);
        allTranslations.push_back(translation);
    }

    if (!markerFound) {
        return;
    }

    Mat firstRot = allRotations[0];
    std::vector<float> firstTrans = allTranslations[0];
    float origPixel[2];
    float xPixel[2];
    float yPixel[2];
    float zPixel[2];

    float xPoint[3] = {firstRot.at<float>(0, 0), firstRot.at<float>(1, 0), firstRot.at<float>(2, 0)};
    float yPoint[3] = {firstRot.at<float>(0, 1), firstRot.at<float>(1, 1), firstRot.at<float>(2, 1)};
    float zPoint[3] = {firstRot.at<float>(0, 2), firstRot.at<float>(1, 2), firstRot.at<float>(2, 2)};
    float translationArr[3] = {firstTrans[0], firstTrans[1], firstTrans[2]};

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

    auto stream = rsCfg.rsProfile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    const struct rs2_intrinsics intrinsics = stream.get_intrinsics();

    rs2::align align(rsCfg.rsAlignTo);
    while(true) {

        rs2::frameset frames = rsCfg.rsPipe.wait_for_frames();
        auto processed = align.process(frames);

        rs2::video_frame other_frame = processed.first_or_default(rsCfg.rsAlignTo);
        rs2::depth_frame depth       = processed.get_depth_frame();

        if (!other_frame || !depth) {
            continue;
        }

        int color_width  = other_frame.get_width();
        int color_height = other_frame.get_height();
        int depth_width  = depth.get_width();
        int depth_height = depth.get_height();

        Mat origFrame(Size (color_width, color_height), CV_8UC3,  (void*)other_frame.get_data(), Mat::AUTO_STEP);
        Mat depthFrame(Size(depth_width, depth_height), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);
        processFrame(origFrame, depthFrame, &intrinsics, rsCfg.depthScale);

        if (displayUi(0, 0, origFrame)) {
            break;
        }
    }
    return 0;
}
