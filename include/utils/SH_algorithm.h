#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

bool inside(cv::Point2d& current_point, cv::Point2d& clip_edge_point1, cv::Point2d& clip_edge_point2);	
double polygonArea(const std::vector<cv::Point2d>& polygon);
void printPolygon(std::vector<cv::Point2d>& );
cv::Point2d intersection(cv::Point2d& , cv::Point2d&, cv::Point2d&, cv::Point2d&);
std::vector<cv::Point2d> SutherlandHodgman(std::vector<cv::Point2d>& subjectPolygon, std::vector<cv::Point2d>& clipPolygon);
// 排列点为顺时针
std::vector<cv::Point2d> sortPointsClockwise(const std::vector<cv::Point2d>& points); 
// conic2polygon
std::vector<cv::Point2d> get_tangent_polygon(double x, double y, double yaw, double a, double b, double extension_length, int num_points);