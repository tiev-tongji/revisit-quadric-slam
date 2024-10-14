/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAP_H
#define MAP_H

#include "QuadricSLAM/MapObject.h"
#include "MapPoint.h"
#include "KeyFrame.h"

#include <set>
#include <vector>
#include <mutex>


namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class MapObject;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void AddMapObject(MapObject* pMO);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void EraseMapObject(MapObject* pMO);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void SetReferenceMapObjects(const std::vector<MapObject*> &vpMOs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    std::vector<MapObject*> GetAllMapObjects();
    std::vector<MapObject*> GetGoodMapObjects();

    void Lock();


    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    std::vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    std::vector<MapObject*> mvpMapObjects;

    double estimated_scale_factor;
    double optimized_scale_factor;
    std::vector<double> estimated_scale_history;
    std::vector<double> optimized_scale_history;

protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;
    std::set<MapObject*> mspMapObjects;

    std::vector<MapPoint*> mvpReferenceMapPoints;
    std::vector<MapObject*> mvpReferenceMapObjects;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
