#include "kabsch.hpp"

#include <array>
#include <opencv2/opencv.hpp>

using namespace cv;

static std::array<float, 3> centerPointCloud(Mat& points)
{
    std::array<float, 3> translation {0, 0, 0};
    int numPoints = points.rows;

    for (int i = 0; i < numPoints; i++) {
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

    return translation;
}

void computeRotAndTrans(Mat &points, Mat &origPoints, std::array<float, 3> &translation, Mat& rot)
{
    translation = centerPointCloud(points);

    Mat refPoints = origPoints.clone();
    std::array<float, 3> pointsTranslation = centerPointCloud(refPoints);
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
    Mat finalTrans = Mat(translation) - rot*Mat(pointsTranslation);

    translation[0] = finalTrans.at<float>(0);
    translation[1] = finalTrans.at<float>(1);
    translation[2] = finalTrans.at<float>(2);
}
