#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() 
{
    const int colorWidth = 1920;
    const int colorHeight = 1080;

    //Enable streams
    rs2::config config;
    config.enable_stream(RS2_STREAM_COLOR, colorWidth, colorHeight, RS2_FORMAT_BGR8, 30);
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    //Begin rs2 pipeline
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start(config);

    //Initialize alignment
    rs2_stream align_to = RS2_STREAM_COLOR;
    rs2::align align(RS2_STREAM_COLOR);

    auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics();

    int xCenter = colorWidth / 2;
    int yCenter = colorHeight / 2;

    int up = yCenter;
    int left = xCenter;
    float vpixel[2];
    float vpoint[3] = {0,0,0};

    while(true) {
        rs2::frameset frames = pipe.wait_for_frames();

        auto processed = align.process(frames);
        rs2::video_frame other_frame = processed.first_or_default(align_to);
        rs2::depth_frame depth = processed.get_depth_frame();

        if (!other_frame || !depth) {
            continue;
        }

        float vdist = depth.get_distance(left, up);
        vpixel[0] = left;
        vpixel[1] = up;
        rs2_deproject_pixel_to_point(vpoint, &intrinsics, vpixel, vdist);

        Mat color(Size(colorWidth, colorHeight), CV_8UC3, (void*)other_frame.get_data(), Mat::AUTO_STEP);
        circle(color, Point(left, up), 4, Scalar(0, 0, 0));

        namedWindow("Show Image", WINDOW_AUTOSIZE);
        imshow("Display Image", color);

        std::cout << left << ", " << up << "[" << vpoint[0] << ", " << vpoint[1] << ", " << vpoint[2] << "]" <<  "\r";

        char button;
        button = waitKey(1);
        if (button == 27) {
            break;
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
    }

    destroyAllWindows();

    return 0;
}
