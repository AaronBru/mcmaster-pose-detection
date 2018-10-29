#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

struct rsConfig {
    rs2::config           rsCfg;
    rs2::align            rsAlign;
    rs2::pipeline         pipe;
    rs2::pipeline_profile profile;

    int colorFps;
    int depthFps;

    std::tuple<int, int>  depthRes;
    std::tuple<int, int>  colorRes;;
};

static bool displayUi(int& left, int& up, Mat& image, std::array<float, 3> vector)
{
    circle(image, Point(left, up), 4, Scalar(0, 0, 0), CV_FILLED);

    std::string displacement = std::to_string(vector[0]) + ", " + std::to_string(vector[1]) + ", " + std::to_string(vector[2]);
    putText(image, displacement, Point(50, 50),  FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 0), 0);

    std::string pixel = std::to_string(left) + ", " + std::to_string(up);
    putText(image, pixel,        Point(50, 100), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 0), 0);

    namedWindow("Show Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    char button;
    button = waitKey(1);
    if (button == 27) {
        return true;
    }
    else if (button == 'w') {
        up += 10;
    }
    else if (button == 's') {
        up -= 10;
    }
    else if (button == 'a') {
        left -= 10;
    }
    else if (button == 'd') {
        left += 10;
    }
    return false;
}


int main() 
{
    const int colorWidth  = 1920;
    const int colorHeight = 1080;

    const int depthWidth  = 640;
    const int depthHeight = 480;

    const int fpsColor = 30;
    const int fpsDepth = 30;

    //Enable streams
    rs2::config config;
    config.enable_stream(RS2_STREAM_COLOR, colorWidth, colorHeight, RS2_FORMAT_BGR8, fpsColor);
    config.enable_stream(RS2_STREAM_DEPTH, depthWidth, depthHeight, RS2_FORMAT_Z16,  fpsDepth);

    //Begin rs2 pipeline
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start(config);

    //Initialize alignment
    rs2_stream align_to = RS2_STREAM_COLOR;
    rs2::align align(RS2_STREAM_COLOR);

    auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics();

    int left = colorWidth / 2;
    int up   = colorHeight / 2;

    float vpixel[2];
    float vPoint[3] = {0,0,0};

    while(true) {
        rs2::frameset frames = pipe.wait_for_frames();

        auto processed = align.process(frames);
        rs2::video_frame other_frame = processed.first_or_default(align_to);
        rs2::depth_frame depth       = processed.get_depth_frame();

        if (!other_frame || !depth) {
            continue;
        }

        vpixel[0] = left;
        vpixel[1] = up;
        rs2_deproject_pixel_to_point(vPoint, &intrinsics, vpixel, depth.get_distance(left, up));

        std::array<float, 3> vec = {vPoint[0], vPoint[1], vPoint[2]};
        Mat color(Size(colorWidth, colorHeight), CV_8UC3, (void*)other_frame.get_data(), Mat::AUTO_STEP);

        if (displayUi(left, up, color,  vec)) {
            break;
        }
    }

    destroyAllWindows();

    return 0;
}
