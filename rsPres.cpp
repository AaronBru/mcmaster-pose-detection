#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;


struct rsConfig {
    rs2::pipeline         rsPipe;
    rs2::pipeline_profile rsProfile;
    rs2_stream            rsAlignTo;

    int colorFps;
    int depthFps;

    std::array<int, 2>  depthRes;
    std::array<int, 2>  colorRes;;
};


static void initRsCam(struct rsConfig &rsCfg)
{
    rsCfg.depthRes = {640, 480};
    rsCfg.colorRes = {1920, 1080};
    rsCfg.colorFps = 30;
    rsCfg.depthFps = 30;

    //Enable streams
    rs2::config config;
    config.enable_stream(RS2_STREAM_COLOR, rsCfg.colorRes[0], rsCfg.colorRes[1], RS2_FORMAT_BGR8, rsCfg.colorFps);
    config.enable_stream(RS2_STREAM_DEPTH, rsCfg.depthRes[0], rsCfg.depthRes[1], RS2_FORMAT_Z16,  rsCfg.depthFps);

    //Begin rs2 pipeline
    rs2::pipeline pipe;
    rsCfg.rsPipe = pipe;
    rsCfg.rsProfile = rsCfg.rsPipe.start(config);

    //Initialize alignment
    rsCfg.rsAlignTo = RS2_STREAM_COLOR;
}


static bool displayUi(int xTarget, int yTarget, Mat& image, std::array<float, 3> vector)
{
    circle(image, Point(xTarget, yTarget), 4, Scalar(0, 0, 0), CV_FILLED);

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

std::array<int, 2> processFrame(Mat& frame)
{
    Mat blur;
    GaussianBlur(frame, blur, cv::Size(5, 5), 3.0, 3.0);
    
    Mat hsvFrame;
    cvtColor(blur, hsvFrame, CV_BGR2HSV);

    Mat rangeRes = Mat::zeros(frame.size(), CV_8UC1);
    inRange(hsvFrame, Scalar(90, 200, 200), Scalar(100, 255, 255), rangeRes);

    dilate(rangeRes, rangeRes, Mat(), Point(-1, -1), 2);

    std::vector<std::vector<Point>> contours;
    findContours(rangeRes, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    std::vector<std::vector<Point>> markers;
    std::vector<Rect> markerBoxes;

    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundBox;
        boundBox = boundingRect(contours[i]);
        
        if (boundBox.area() > 400) {
            markers.push_back(contours[i]);
            markerBoxes.push_back(boundBox);
        }
    }

    Point maxCenter;
    int maxArea = 0;
    for (size_t i = 0; i < markers.size(); i++) {

        drawContours(frame, markers, i,  CV_RGB(150, 20, 20), 3);
        rectangle(frame, markerBoxes[i], CV_RGB(20, 150, 20), 3);

        Point center;
        center.x = markerBoxes[i].x + markerBoxes[i].width / 2;
        center.y = markerBoxes[i].y + markerBoxes[i].height / 2;

        int currentArea = markerBoxes[i].area();
        if (currentArea > maxArea) {
            maxArea = currentArea;
            maxCenter.x = center.x;
            maxCenter.y = center.y;
        }

        circle(frame, center, 4, CV_RGB(20, 150, 20), CV_FILLED);
    }

    std::array<int, 2> markerCenter;
    if (markers.size() == 0) {
        markerCenter[0] = 100;
        markerCenter[1] = 100;
    } 
    else {
        markerCenter[0] = maxCenter.x;
        markerCenter[1] = maxCenter.y;
    }
    return markerCenter;
}


int main() 
{
    struct rsConfig rsCfg;
    initRsCam(rsCfg);

    auto stream = rsCfg.rsProfile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics();

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

        Mat color(Size(rsCfg.colorRes[0], rsCfg.colorRes[1]), CV_8UC3, (void*)other_frame.get_data(), Mat::AUTO_STEP);
        std::array<int, 2> trackPos = processFrame(color);

        vPixel[0] = trackPos[0];
        vPixel[1] = trackPos[1];
        rs2_deproject_pixel_to_point(vPoint, &intrinsics, vPixel, depth.get_distance(trackPos[0], trackPos[1]));

        std::array<float, 3> vec = {vPoint[0], vPoint[1], vPoint[2]};
        if (displayUi(trackPos[0], trackPos[1] , color,  vec)) {
            break;
        }
    }
    return 0;
}
