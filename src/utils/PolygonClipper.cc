#include "utils/PolygonClipper.h"

// 构造函数，初始化警告选项
PolygonClipper::PolygonClipper(bool warn_if_empty) : warn_if_empty(warn_if_empty) {}

// 检查点是否在边界内部
bool PolygonClipper::is_inside(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& q) {
    double R = (p2.x - p1.x) * (q.y - p1.y) - (p2.y - p1.y) * (q.x - p1.x);
    return R <= 0;
}

// 计算两条线段的交点
cv::Point2d PolygonClipper::compute_intersection(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& p3, const cv::Point2d& p4) {
    double x, y;

    if (p2.x - p1.x == 0) {
        x = p1.x;
        double m2 = (p4.y - p3.y) / (p4.x - p3.x);
        double b2 = p3.y - m2 * p3.x;
        y = m2 * x + b2;
    } else if (p4.x - p3.x == 0) {
        x = p3.x;
        double m1 = (p2.y - p1.y) / (p2.x - p1.x);
        double b1 = p1.y - m1 * p1.x;
        y = m1 * x + b1;
    } else {
        double m1 = (p2.y - p1.y) / (p2.x - p1.x);
        double b1 = p1.y - m1 * p1.x;
        double m2 = (p4.y - p3.y) / (p4.x - p3.x);
        double b2 = p3.y - m2 * p3.x;
        x = (b2 - b1) / (m1 - m2);
        y = m1 * x + b1;
    }

    return { x, y };
}


// 裁剪函数，返回裁剪后的多边形顶点
std::vector<cv::Point2d> PolygonClipper::clip(const std::vector<cv::Point2d>& subject_polygon, const std::vector<cv::Point2d>& clipping_polygon) {
    std::vector<cv::Point2d> final_polygon = subject_polygon;

    for (size_t i = 0; i < clipping_polygon.size(); ++i) {
        std::vector<cv::Point2d> next_polygon = final_polygon;
        final_polygon.clear();

        cv::Point2d c_edge_start = clipping_polygon[i];
        cv::Point2d c_edge_end = clipping_polygon[(i + 1) % clipping_polygon.size()];

        for (size_t j = 0; j < next_polygon.size(); ++j) {
            cv::Point2d s_edge_start = next_polygon[j];
            cv::Point2d s_edge_end = next_polygon[(j + 1) % next_polygon.size()];

            if (is_inside(c_edge_start, c_edge_end, s_edge_end)) {
                if (!is_inside(c_edge_start, c_edge_end, s_edge_start)) {
                    final_polygon.push_back(compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end));
                }
                final_polygon.push_back(s_edge_end);
            } else if (is_inside(c_edge_start, c_edge_end, s_edge_start)) {
                final_polygon.push_back(compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end));
            }
        }
    }

    if (final_polygon.empty() && warn_if_empty) {
        // std::cerr << "No intersections found. Are you sure your polygon coordinates are in clockwise order?" << std::endl;
    }

    return final_polygon;
}
