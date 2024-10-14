#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "utils/SH_algorithm.h"
#include "utils/PolygonClipper.h"

typedef Eigen::Matrix<double, 5, 1> Vector5d;

// bbox IoU
float RectIoU(const cv::Rect &rect1, const cv::Rect &rect2);

float RectIoUFormer(const cv::Rect& rect1, const cv::Rect& rect2);

float RectIoULatter(const cv::Rect& rect1, const cv::Rect& rect2);

double GetMeanConicPolygonIoU(Vector5d conic, std::vector<cv::Point> convex_hull_points);

double logit_inverse(const int num);

// line point, line direction vector, normal-form plane
bool line_plane_intersection(Eigen::Vector3d& line_p, 
                             Eigen::Vector3d& line_direction, 
                             Eigen::Vector4d& plane,
                             Eigen::Vector3d& intersction);

#endif