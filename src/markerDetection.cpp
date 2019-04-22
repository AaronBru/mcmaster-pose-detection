/********************************************************************************

Example application illustrating use of this library to detect pose of an object.

*********************************************************************************/

#include "aruco.hpp"
#include "rsCam.hpp"
#include "detectPoseRealsense.hpp"

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

using namespace cv;

#define POINTS_PER_MARKER 4
#define FACE_ID 0
#define SIDE_ID 5
#define TOP_ID  11
std::vector<int> validIds {FACE_ID, SIDE_ID, TOP_ID};

#define OBJECT_HEIGHT 0.14
#define OBJECT_WIDTH  0.078
#define OBJECT_DEPTH  0.068

#define MARKER_WIDTH 0.05

/* Set up positions of corners on object with respect to Aruco markers. Corners are referenced clockwise, 
beginning from the top left corner. */
static float facePoints[POINTS_PER_MARKER][3] =     {{-MARKER_WIDTH / 2,  MARKER_WIDTH / 2, OBJECT_DEPTH / 2},
                                                     { MARKER_WIDTH / 2,  MARKER_WIDTH / 2, OBJECT_DEPTH / 2},
                                                     { MARKER_WIDTH / 2, -MARKER_WIDTH / 2, OBJECT_DEPTH / 2},
                                                     {-MARKER_WIDTH / 2, -MARKER_WIDTH / 2, OBJECT_DEPTH / 2}};

static float sidePoints[POINTS_PER_MARKER][3] =     {{ OBJECT_WIDTH / 2,  MARKER_WIDTH / 2, MARKER_WIDTH / 2},
                                                     { OBJECT_WIDTH / 2,  MARKER_WIDTH / 2, -MARKER_WIDTH / 2},
                                                     { OBJECT_WIDTH / 2, -MARKER_WIDTH / 2, -MARKER_WIDTH / 2},
                                                     { OBJECT_WIDTH / 2, -MARKER_WIDTH / 2, MARKER_WIDTH / 2}};

static float topPoints[POINTS_PER_MARKER][3]  =     {{-MARKER_WIDTH / 2,  OBJECT_HEIGHT / 2, -MARKER_WIDTH / 2},
                                                     { MARKER_WIDTH / 2,  OBJECT_HEIGHT / 2, -MARKER_WIDTH / 2},
                                                     { MARKER_WIDTH / 2,  OBJECT_HEIGHT / 2, MARKER_WIDTH / 2},
                                                     {-MARKER_WIDTH / 2,  OBJECT_HEIGHT / 2, MARKER_WIDTH / 2}};

static const Mat faceMarker(POINTS_PER_MARKER, 3, CV_32FC1, facePoints);
static const Mat sideMarker(POINTS_PER_MARKER, 3, CV_32FC1, sidePoints);
static const Mat topMarker (POINTS_PER_MARKER, 3, CV_32FC1, topPoints);

static       std::map<int, Mat> objectMarker = {{FACE_ID, faceMarker},
                                                 {SIDE_ID, sideMarker},
                                                 {TOP_ID,  topMarker}};

/* Helper function to display window. Hit escape to close application */
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

int main() 
{
    struct rsConfig rsCfg;

    /* Set up parameters.  Note this application does not use IR, but only depth and color */
    rsCfg.depthRes = {640, 480};
    rsCfg.irRes    = {640, 480};
    rsCfg.colorRes = {640, 480};
    rsCfg.colorFps = 60;
    rsCfg.irFps    = 60;
    rsCfg.depthFps = 60;

    /* Initialize camera */
    initRsCam(rsCfg);

    auto stream = rsCfg.rsProfile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    const struct rs2_intrinsics intrinsics = stream.get_intrinsics();

    rs2::frame_queue frameQueue(5);
    std::atomic_bool alive {true};

    /* This thread used solely to receive frames and check if color and depth frames are valid */
    std::thread rxFrame([&]() {
        while(alive) {

            rs2::frameset frames = rsCfg.rsPipe.wait_for_frames();

            auto colorFrame = frames.get_color_frame();
            auto depthFrame = frames.get_depth_frame();

            if (!colorFrame || !depthFrame) {
                continue;
            }
            frameQueue.enqueue(frames);    
        }
    });

    rs2::frameset curFrame;
    auto start = std::chrono::high_resolution_clock::now();
    char frameRate[10];

    /* Note that validIds is a vector of all ids that are searched among ALL objects */
    DetectPoseRealsense poseDetect(&intrinsics, rsCfg.depthScale, validIds);
    Mat rot(Size(3,3), CV_32FC1);
    std::array<float, 3> translation {0, 0, 0};

    rs2::align align(rsCfg.rsAlignTo);
    while(alive) {
        /* Receive frames from other thread here */
        frameQueue.poll_for_frame(&curFrame);

        if (curFrame) {
            auto processed = align.process(curFrame);

            rs2::video_frame other_frame = processed.first_or_default(rsCfg.rsAlignTo);
            rs2::depth_frame depth       = processed.get_depth_frame();

            int color_width  = other_frame.get_width();
            int color_height = other_frame.get_height();
            int depth_width  = depth.get_width();
            int depth_height = depth.get_height();

            Mat origFrame(Size(color_width, color_height), CV_8UC3,  (void*)other_frame.get_data(), Mat::AUTO_STEP);
            Mat depthFrame(Size(depth_width, depth_height), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);
            poseDetect.processFrame(origFrame, depthFrame);

            /* Detect pose here with the object map created above */
            if (poseDetect.getPose(objectMarker, rot, translation)) {
                drawMarker(rot, translation, &intrinsics, origFrame);
            }

            /* Determine latency in milliseconds between frames */
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            float milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            printf("TIME: %02f\n", milliseconds);
            snprintf(frameRate, sizeof(frameRate), "%02f\n", milliseconds);
            putText(origFrame, frameRate, Point(50, 50), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 3);

            start = std::chrono::high_resolution_clock::now();
            if (displayUi(0, 0, origFrame)) {

                /* Signal to threads to end */
                alive = false;
            }
        }
    }

    rxFrame.join();
    return 0;
}
