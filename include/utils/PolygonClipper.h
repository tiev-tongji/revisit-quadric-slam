#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// 多边形裁剪类
class PolygonClipper {
public:
    // 构造函数，初始化警告选项
    PolygonClipper(bool warn_if_empty = true);

    // 裁剪函数，返回裁剪后的多边形顶点
    std::vector<cv::Point2d> clip(const std::vector<cv::Point2d>& subject_polygon, const std::vector<cv::Point2d>& clipping_polygon);

private:
    // 检查点是否在边界内部
    bool is_inside(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& q);

    // 计算两条线段的交点
    cv::Point2d compute_intersection(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& p3, const cv::Point2d& p4);

    bool warn_if_empty;
};
