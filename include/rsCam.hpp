#ifndef _RSCAM_HPP
#define _RSCAM_HPP

#include <librealsense2/rs.hpp>

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

void initRsCam(struct rsConfig &rsCfg);
float getDepthScale(rs2::device dev);

#endif
