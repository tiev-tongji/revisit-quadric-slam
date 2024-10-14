#include<utils/utils.h>
#include<cmath>
// #include<opencv2/core/core.hpp>

// bbox IoU
float RectIoU(const cv::Rect &rect1, const cv::Rect &rect2)
{
    int overlap_area = (rect1 & rect2).area();
    float union_area = (float)rect1.area() + (float)rect2.area() - (float)overlap_area;
    return (float)overlap_area / union_area;
}

float RectIoUFormer(const cv::Rect& rect1, const cv::Rect& rect2)
{
    int overlap_area = (rect1&rect2).area();
    return (float)overlap_area/((float)(rect1.area()));
}

float RectIoULatter(const cv::Rect& rect1, const cv::Rect& rect2)
{
    int overlap_area = (rect1&rect2).area();
    return (float)overlap_area/((float)(rect2.area()));
}

double GetMeanConicPolygonIoU(Vector5d conic, std::vector<cv::Point> convex_hull_points){
    double iou = 0;
    for(int i=0;i<conic.size();i++){
        if(std::isnan(conic[i])){
            return 0;
        }
    }
    // 使用 Sutherland-Hodgman 算法进行裁剪
    std::vector<cv::Point2d> subjectPolygonVertices = get_tangent_polygon(conic[0],
                                                                        conic[1],
                                                                        conic[2],
                                                                        conic[3],
                                                                        conic[4],
                                                                        0.1, 10);
    std::vector<cv::Point2d> clipPolygonVertices;
    // Point to Point2d
    for(int i=0;i<convex_hull_points.size();i++){
        clipPolygonVertices.push_back(cv::Point2d(convex_hull_points[i].x, convex_hull_points[i].y));
    }

    subjectPolygonVertices = sortPointsClockwise(subjectPolygonVertices);
    clipPolygonVertices = sortPointsClockwise(clipPolygonVertices);

    PolygonClipper clipper;
    std::vector<cv::Point2d> resultPolygonVertices = clipper.clip(subjectPolygonVertices, clipPolygonVertices);

    // 计算面积
    double areaSubject = polygonArea(subjectPolygonVertices);
    double areaClip = polygonArea(clipPolygonVertices);
    double areaIntersection = polygonArea(resultPolygonVertices);
    // 计算 IoU
    // std::cout<<areaIntersection<<std::endl;
    if((areaSubject + areaClip - areaIntersection)==0){
        return 0;
    }
    iou = areaIntersection / (areaSubject + areaClip - areaIntersection);
    // std::cout << "IoU: " << iou << std::endl;
    return iou;

}

double logit_inverse(const int x)
{
    return exp((double)x)/(exp((double)x)+1);
}

// line point, line direction vector, normal-form plane
bool line_plane_intersection(Eigen::Vector3d& line_point, 
                             Eigen::Vector3d& line_direction, 
                             Eigen::Vector4d& plane,
                             Eigen::Vector3d& intersection)
{
    // normlize the plane
    plane = plane/plane.head<3>().norm();
    Eigen::Vector3d plane_normal = plane.head<3>();
    double d = plane[3];
    Eigen::Vector3d plane_point = plane_normal * (-d/plane_normal.squaredNorm());

    if(fabs(plane_normal.dot(line_direction)) > 1e-5)
    {
        double t = (plane_normal.dot(plane_point)-plane_normal.dot(line_point))/plane_normal.dot(line_direction.normalized());
        intersection = line_point+line_direction.normalized()*t;
        return true;
    }
    else
    {
        return false;
    }

}