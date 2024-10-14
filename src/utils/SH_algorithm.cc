#include "utils/SH_algorithm.h"
#include <iostream>
#include <algorithm> // 需要包含这个头文件来使用 std::sort
#include <cmath>     // 需要包含这个头文件来使用 std::atan2

void printPolygon(std::vector<cv::Point2d>& polygon){
	for(int i =0; i < polygon.size(); i++)
		std::cout << "( " << polygon[i].x << " , " << polygon[i].y << " )" <<"\n"; 
		std::cout << "\n";
}


bool inside(cv::Point2d& point, cv::Point2d& clip_edge_point1, cv::Point2d& clip_edge_point2)
{
    return (clip_edge_point2.y - clip_edge_point1.y) * point.x + (clip_edge_point1.x - clip_edge_point2.x) * point.y + (clip_edge_point2.x * clip_edge_point1.y - clip_edge_point1.x * clip_edge_point2.y) < 0;
}

// 计算多边形面积
double polygonArea(const std::vector<cv::Point2d>& polygon) {
    double area = 0.0;
    int n = polygon.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
    }
    return std::abs(area) / 2.0;
}

std::vector<cv::Point2d> sortPointsClockwise(const std::vector<cv::Point2d>& points) {
    cv::Point2d center = {0.0, 0.0};
    for (const auto& point : points) {
        center.x += point.x;
        center.y += point.y;
    }
    center.x /= points.size();
    center.y /= points.size();

    std::vector<cv::Point2d> sortedPoints = points;

    std::sort(sortedPoints.begin(), sortedPoints.end(), [&](const cv::Point2d& a, const cv::Point2d& b) {
        return std::atan2(a.y - center.y, a.x - center.x) > std::atan2(b.y - center.y, b.x - center.x);
    });

    return sortedPoints;
}

cv::Point2d intersection(cv::Point2d& edge_point1, cv::Point2d& edge_point2, cv::Point2d& prev_point, cv::Point2d& current_point)
{
    cv::Point2d dc = { edge_point1.x - edge_point2.x, edge_point1.y - edge_point2.y };
    cv::Point2d dp = { prev_point.x - current_point.x, prev_point.y - current_point.y };
 
    float n1 = edge_point1.x * edge_point2.y - edge_point1.y * edge_point2.x;
    float n2 = prev_point.x * current_point.y - prev_point.y * current_point.x;
    float n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x);
 
    return { (n1 * dp.x - n2 * dc.x) * n3, (n1 * dp.y - n2 * dc.y) * n3 };
}

std::vector<cv::Point2d> get_tangent_polygon(double x, double y, double yaw, double a, double b, double extension_length = 0.1, int num_points = 10) {
    std::vector<cv::Point2d> vertices;
    double angle_step = 2 * M_PI / num_points;
    
    for (int i = 0; i < num_points; ++i) {
        double theta = i * angle_step;
        double xt = x + a * cos(theta) * cos(yaw) - b * sin(theta) * sin(yaw);
        double yt = y + a * cos(theta) * sin(yaw) + b * sin(theta) * cos(yaw);
        vertices.emplace_back(cv::Point2d(xt + extension_length * cos(theta), yt + extension_length * sin(theta)));
    }
    
    return vertices;
}


/*
* The below code implements the SutherlandHodgman algorithm. It's the exact implementation of 
* the pseudocode of wikipedia link : https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
*
*/
std::vector<cv::Point2d> SutherlandHodgman(std::vector<cv::Point2d>& subjectPolygon, std::vector<cv::Point2d>& clipPolygon)
{
		std::vector<cv::Point2d> outputList(subjectPolygon);
		for(int edge =0; edge< clipPolygon.size();edge++){
			cv::Point2d edge_1_point = clipPolygon[edge];
			cv::Point2d edge_2_point = clipPolygon[(edge+1)%clipPolygon.size()];
			std::vector<cv::Point2d> inputList(outputList);
			outputList.clear();
			
			for(int i =0; i<inputList.size();i++){
				cv::Point2d current_point = inputList[i];
				cv::Point2d prev_point = inputList[(i+inputList.size()-1)%inputList.size()];
				
				//cv::Point2d intersecting_point = intersection()
				
				//intersection(cv::Point2d cp1, cv::Point2d cp2, cv::Point2d s, cv::Point2d e)
				if(inside(current_point,edge_1_point,edge_2_point)){
					if(!inside(prev_point,edge_1_point,edge_2_point)){
						outputList.push_back(intersection(edge_1_point, edge_2_point, prev_point, current_point));
					}
					outputList.push_back(current_point);
				}else if(inside(prev_point,edge_1_point, edge_2_point)){
					outputList.push_back(intersection(edge_1_point, edge_2_point, prev_point, current_point));
				}
				
			}
		}
		return outputList;
		
}
	
	