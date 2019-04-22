#include "rsCam.hpp"

#include <librealsense2/rs.hpp>

float getDepthScale(rs2::device dev)
{
    for (rs2::sensor& sensor : dev.query_sensors()) {
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>()) {
            return dpt.get_depth_scale();
        }
    }
    return 1.0;
}

void initRsCam(struct rsConfig &rsCfg)
{
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

