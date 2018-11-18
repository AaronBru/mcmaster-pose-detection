#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;


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
    //config.enable_stream(RS2_STREAM_COLOR, rsCfg.colorRes[0], rsCfg.colorRes[1], RS2_FORMAT_BGR8, rsCfg.colorFps);
    config.enable_stream(RS2_STREAM_INFRARED, rsCfg.irRes[0], rsCfg.irRes[1], RS2_FORMAT_Y8, rsCfg.irFps);
    config.enable_stream(RS2_STREAM_DEPTH, rsCfg.depthRes[0], rsCfg.depthRes[1], RS2_FORMAT_Z16,  rsCfg.depthFps);

    //Begin rs2 pipeline
    rs2::pipeline pipe;
    rsCfg.rsPipe = pipe;
    rsCfg.rsProfile = rsCfg.rsPipe.start(config);

    //Initialize alignment
    rsCfg.rsAlignTo = RS2_STREAM_INFRARED;
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
    return std::array<int, 2> {0, 0};
}


int main() 
{
    struct rsConfig rsCfg;
    initRsCam(rsCfg);

    auto stream = rsCfg.rsProfile.get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
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

        Mat origFrame(Size(rsCfg.irRes[0], rsCfg.irRes[1]), CV_8UC1, (void*)other_frame.get_data(), Mat::AUTO_STEP);
        std::array<int, 2> trackPos = processFrame(origFrame);

        vPixel[0] = trackPos[0];
        vPixel[1] = trackPos[1];
        rs2_deproject_pixel_to_point(vPoint, &intrinsics, vPixel, depth.get_distance(trackPos[0], trackPos[1]));

        std::array<float, 3> vec = {vPoint[0], vPoint[1], vPoint[2]};
        if (displayUi(trackPos[0], trackPos[1] , origFrame,  vec)) {
            break;
        }
    }
    return 0;
}
