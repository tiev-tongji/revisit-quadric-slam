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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"QuadricSLAM/Parameters.h"
#include"utils/utils.h"
#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>
#include<fstream>
#include<sstream>
#include<iterator>
#include<mutex>




using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0), mpQuadricSolver(nullptr)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // 自己添加的，获取img的width和height
    int img_width = fSettings["Camera.width"];
    int img_height = fSettings["Camera.height"];

    if((mask = imread("./masks/mask.png", cv::IMREAD_GRAYSCALE)).empty())
        mask = cv::Mat();

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImRaw = im.clone();
    mImGray = im.clone();
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
    // 特征点和特征线构成当前帧
    
    imwrite("./mask.png", mask);

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mImRaw,mask);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mImRaw,mask);
    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    // Step 1：地图初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            //单目初始化
            MonocularInitialization();
        // 画图
        mpFrameDrawer->Update(this);

        //这个状态量在上面的初始化函数中被更新
        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        // bOK为临时变量，用于表示每个函数是否执行成功
        bool bOK;

        // tracking 类构造时默认为false。在viewer中有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking
        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            
            // Step 2：跟踪进入正常SLAM模式，有地图更新
            // 是否正常跟踪
            if(mState==OK)
            {
                
                // Local Mapping might have changed some MapPoints & MapLines tracked in last frame
                // Step 2.1 检查并更新上一帧被替换的MapPoints
                // 局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();

                // Step 2.2 运动模型是空的或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
                // 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
                // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
                // mnLastRelocFrameId 上一次重定位的那一帧
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // 用最近的关键帧来跟踪当前的普通帧
                    // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点都对应3D点重投影误差即可得到位姿
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                    // 用最近的普通帧来跟踪当前的普通帧
                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        //根据恒速模型失败了，只能根据参考关键帧来跟踪
                        bOK = TrackReferenceKeyFrame();
                    
                    //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                    //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
                    //cout << "Track time: " << time_used.count() << endl;
                }
            }
            else
            {
                // 如果跟踪状态不成功,那么就只能重定位了
                // BOW搜索，EPnP求解位姿
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated
            // Step 2：只进行跟踪tracking，局部地图不工作
            if(mState==LOST)
            {
                // Step 2.1 如果跟丢了，只能重定位
                bOK = Relocalization();
            }
            else
            {
                // mbVO是mbOnlyTracking为true时的才有的一个变量
                // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常 (注意有点反直觉)
                // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints & MapLines in the map
                    // Step 2.2 如果跟踪正常，使用恒速模型 或 参考关键帧跟踪
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        // 如果恒速模型不被满足,那么就只能够通过参考关键帧来定位
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    // mbVO为true，表明此帧匹配了很少（小于10）的地图点，要跪的节奏，既做跟踪又做重定位

                    //MM=Motion Model,通过运动模型进行跟踪的结果
                    bool bOKMM = false;
                    //通过重定位方法来跟踪的结果
                    bool bOKReloc = false;

                    //运动模型中构造的地图点
                    vector<MapPoint*> vpMPsMM;
                    //在追踪运动模型后发现的外点
                    vector<bool> vbOutMM;
                    //运动模型得到的位姿
                    cv::Mat TcwMM;

                    // Step 2.3 当运动模型有效的时候,根据运动模型计算位姿
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();

                        // 将恒速模型跟踪结果暂存到这几个变量中，因为后面重定位会改变这些变量
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    // Step 2.4 使用重定位的方法来得到当前帧的位姿
                    bOKReloc = Relocalization();

                    // Step 2.5 根据前面的恒速模型、重定位结果来更新状态
                    if(bOKMM && !bOKReloc)
                    {
                        // 恒速模型成功、重定位失败，重新使用之前暂存的恒速模型结果
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        // 如果当前帧匹配的3D点很少，增加当前可视地图点的被观测次数
                        if(mbVO)
                        {
                            // 更新当前帧的地图点被观测次数
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                //如果这个特征点形成了地图点,并且也不是外点的时候
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    //增加能观测到该地图点的帧数
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        // 只要重定位成功整个跟踪过程正常进行（重定位与跟踪，更相信重定位）
                        mbVO = false;
                    }
                    //有一个成功我们就认为执行成功了
                    bOK = bOKReloc || bOKMM;
                }
            }
        }
        
        // 将最新的关键帧作为当前帧的参考关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // Step 3：在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        // 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            // 重定位成功
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }
        //根据上面的操作来判断是否追踪成功
        if(bOK)
            mState = OK;
        else
            mState=LOST;
        // Update drawer
        // Step 4：更新显示线程中的图像、特征点、地图点等信息
        mpFrameDrawer->Update(this);
        // If tracking were good, check if we insert a keyframe
        //只有在成功追踪时才考虑生成关键帧的问题
        if(bOK)
        {
            // Update motion model
            // Step 5：跟踪成功，更新恒速运动模型
            if(!mLastFrame.mTcw.empty())
            {
                // 更新恒速运动模型 TrackWithMotionModel 中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换， 其中 Twl = LastTwc
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                //否则速度为空  
                mVelocity = cv::Mat();

            //更新显示中的位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            // Step 6：清除观测不到的地图点,地图线
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }


            // Delete temporal MapPoints
            // Step 7：清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
            // 步骤6中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            // 不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
            mlpTemporalPoints.clear();
            // Check if we need to insert a new keyframe
            // Step 8：检测并插入关键帧，对于双目或RGB-D会产生新的地图点
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();
            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            //  Step 9 删除那些在bundle adjustment中检测为outlier的地图点和地图线   
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                // 这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
            
        }
        // Reset if the camera get lost soon after initialization
        // Step 10 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
        if(mState==LOST)
        {
            //如果地图中的关键帧信息过少的话,直接重新进行初始化了
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }
        //确保已经设置了参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        
        // 保存上一帧的数据,当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
    }
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // Step 11：记录位姿信息，用于最后保存所有的轨迹
    if(!mCurrentFrame.mTcw.empty())
    {
        // 计算相对姿态Tcr = Tcw * Twr, Twr = Trw^-1
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        //保存各种状态
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        // 如果跟踪失败，则相对位姿使用上一次值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            mbIniFirst = false;

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)  

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            // 删除那些无法三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }
            
            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
        // line?
        // mLastFrame = Frame(mCurrentFrame);
    }
}

// 为单目摄像头三角化生成MapPoints，只有特征点
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<80)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/*
 * @brief 用参考关键帧的地图点来对当前普通帧进行跟踪
 * 
 * Step 1：将当前普通帧的描述子转化为BoW向量
 * Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
 * Step 3: 将上一帧的位姿态作为当前帧位姿的初始值
 * Step 4: 通过优化3D-2D的重投影误差来获得位姿
 * Step 5：剔除优化后的匹配点中的外点
 * @return 如果匹配数超10，返回true
 * 
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    
    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard point outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

/**
 * @brief 根据匀速模型对上一帧的MapPoints进行跟踪
 *
 * 1.非单目情况，需要对上一帧产生一些新的MapPoints(临时)
 * 2.将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
 * 3.根据匹配对估计当前帧的姿态
 * 4.根据姿态剔除误匹配
 * @return 如果匹配数大于10，则返回true
 */
bool Tracking::TrackWithMotionModel()
{
    // 建立ORB特征点的匹配
    // 最小距离 < 0.9*次小距离 匹配成功，检查旋转
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
    UpdateLastFrame();

    // Step 2：根据之前估计的速度，用恒速模型得到当前帧的初始位姿。
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    // quadric slam for visualization
    DrawQuadricProject(mCurrentFrame.mQuadricImage,true);

    // 清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;//单目
    else
        th=7;//双目

    // Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    // 如果匹配点太少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    // 如果还是不能够获得足够的匹配点,那么就认为跟踪失败
    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    // Step 4：利用3D-2D投影关系，优化当前帧位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard point outliers
    // Step 5：剔除地图点中外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                // 累加成功匹配到的地图点数目
                nmatchesMap++;
        }
    }
    

    if(mbOnlyTracking)
    {
        // 纯定位模式下：如果成功追踪的地图点非常少,那么这里的mbVO标志就会置位
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }
    
    // Step 6：匹配超过10个点就认为跟踪成功
    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();
    SearchLocalPoints();

    associateObjectsInFrame(mCurrentFrame);

    // Optimize Pose
    Optimizer::PoseJointOptimization(&mCurrentFrame);

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);
    DetectSegmentationOffline(pKF); // 读取语义分割结果
    AssociateObjets(pKF);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::DetectObjectOffline(KeyFrame* pKF)
{
    char filename[256];
    std::string bbox_file_path;
    bbox_file_path = mpSystem->mstrSequence + "/yolo_txt/";

    sprintf(filename, "%s%06lu.txt", bbox_file_path.c_str(), pKF->mnFrameId);
    Eigen::MatrixXd detection_mat, detection_mat_trans;
    read_all_number_txt(std::string(filename), detection_mat);

    detection_mat_trans = detection_mat.transpose();
    for(int i=0; i<detection_mat_trans.cols(); i++)
    {
        Eigen::VectorXd detection;
        detection.resize(8);
        detection << detection_mat_trans.col(i);
        Object_2D* obj = new Object_2D(detection);
        obj->DetermineScaleLevel(pKF->mnRows, pKF->mnCols);
        // obj->PrintInfo();
        pKF->mvpLocalObjects_2D.push_back(obj);
    }
    //printf("detect 2d object offline: %d  frame id: %lu \n",pKF->mvpLocalObjects_2D.size(), pKF->mnFrameId);
    int N = pKF->mvpLocalObjects_2D.size();
    // 3D object matches
    pKF->mvpLocalMapObjects = vector<MapObject*>(N,static_cast<MapObject*>(NULL));
}

void Tracking::DetectSegmentationOffline(KeyFrame* pKF)
{
    char filename[256];
    std::string bbox_file_path;
    bbox_file_path = mpSystem->mstrSequence + "/segmentation_txt/";
    sprintf(filename, "%s%06lu.txt", bbox_file_path.c_str(), pKF->mnFrameId);
    Eigen::MatrixXd detection_mat, detection_mat_trans;

    std::ifstream file(filename);
    if(!file.is_open())
    {
        std::cout << "ERROR!!! Cannot read txt file " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            std::istringstream iss(line);
            int instance_id;
            int track_id;
            int label;
            double confidence;

            std::vector<cv::Point> contour_points;

            // 读取 index、instance、id、label 和 confidence
            iss >> instance_id >> track_id >> label >> confidence;
            // iss >> instance_id >> label >> confidence;
            label = label + 1;

            double x, y;
            cv::Point p;
            while (iss >> p.x >> p.y) { // 读取 contour points
                contour_points.push_back(p);
            }
            // if(confidence < 0.90){
            //     continue;
            // }

            // contours bounding box
            int xmin;
            int ymin;
            int xmax;
            int ymax;

            if (contour_points.empty()) {
                continue;
            }

            // 初始化边界框的 xmin、ymin、xmax 和 ymax
            xmin = xmax = contour_points[0].x;
            ymin = ymax = contour_points[0].y;

            // 计算轮廓点中的最小和最大坐标
            for (const auto& point : contour_points) {
                xmin = std::min(xmin, point.x);
                ymin = std::min(ymin, point.y);
                xmax = std::max(xmax, point.x);
                ymax = std::max(ymax, point.y);
            }

            Eigen::VectorXd detection;
            detection.resize(8);
            detection.array() << instance_id, xmin, ymin, xmax, ymax, confidence, label;
            
            Object_2D* obj = new Object_2D(detection);
            obj->DetermineScaleLevel(pKF->mnRows, pKF->mnCols);
            obj->contour_points = contour_points;

            // obj->convex_hull_lines = getLinesFromContour(contour_points);
            getLinesFromContour(contour_points, obj->convex_hull_lines, obj->convex_hull_points);
            // obj->PrintInfo();
            pKF->mvpLocalObjects_2D.push_back(obj);

        }
    }

    file.close();
    //printf("detect 2d object offline: %d  frame id: %lu \n",pKF->mvpLocalObjects_2D.size(), pKF->mnFrameId);
    int N = pKF->mvpLocalObjects_2D.size();
    // 3D object matches
    pKF->mvpLocalMapObjects = vector<MapObject*>(N,static_cast<MapObject*>(NULL));
}

// contour points->convex hull->line simplification->bad line culling
void Tracking::getLinesFromContour(std::vector<cv::Point> contour_points, std::vector<Eigen::Vector3d> &convex_hull_lines, std::vector<cv::Point> &convex_hull_points){
    // 转换轮廓点为OpenCV需要的数据类型
    std::vector<Point> hull;
    cv::convexHull(contour_points, hull);

    // Douglas-Peucker算法对凸包点进行线段简化
    std::vector<Point> simplified_hull;
    double epsilon = 3.0; // 设置Douglas-Peucker算法的阈值
    cv::approxPolyDP(hull, simplified_hull, epsilon, true);

    // 将简化后的线段表示为ax + by + c = 0的形式，并存储为Vector3d
    std::vector<Vector3d> lines;
    for (size_t i = 0; i < simplified_hull.size(); ++i) {
        int next_idx = (i + 1) % simplified_hull.size();
        Point pt1 = simplified_hull[i];
        Point pt2 = simplified_hull[next_idx];

        // 计算直线参数a、b、c
        double a = pt2.y - pt1.y;
        double b = pt1.x - pt2.x;
        double c = pt2.x * pt1.y - pt1.x * pt2.y;

        // 归一化直线参数
        double norm_factor = sqrt(a * a + b * b);
        a /= norm_factor;
        b /= norm_factor;
        c /= norm_factor;

        // 存储到Vector3d对象中并添加到结果向量中
        lines.push_back(Vector3d(a, b, c));
    }
    convex_hull_lines = lines;
    convex_hull_points = simplified_hull;
}

void Tracking::AssociateObjectMapPoint(KeyFrame* pKF)
{
    // associate map points with 2d object
    for(auto obj_2d : pKF->mvpLocalObjects_2D)
    {
        // 创建一个与轮廓所在图像相同大小的二值图像
        cv::Mat mask = cv::Mat::zeros(mImRaw.rows, mImRaw.cols, CV_8UC1);
        std::vector<std::vector<cv::Point>> contours = {obj_2d->contour_points};
        cv::drawContours(mask, contours, 0, cv::Scalar(255), cv::FILLED);

        int segmentation_points_num = 0;
        int bbox_points_num = 0;
        for(size_t i = 0; i < pKF->GetMapPointMatches().size(); i++)
        {
            MapPoint *pMP = pKF->GetMapPoint(i);
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;

            if (mask.at<uchar>(pKF->mvKeysUn[i].pt) > 0) {
                cv::Mat PointPosWorld = pMP->GetWorldPos();                 // world frame.
                pMP->object_view = true;                  // the point is associated with an object.
                pMP->frame_id.insert(pKF->mnId); // no use.
                // pMP->mfeature = pKF->mvKeysUn[i]; // seems no use as well
                // object points.
                obj_2d->Obj_c_MapPonits.push_back(pMP);
                // summation the position of points.
                obj_2d->sum_pos_3d += PointPosWorld;
                segmentation_points_num++;
            }

            if(obj_2d->mBoxRect.contains(pKF->mvKeysUn[i].pt))// in rect.
            {
                bbox_points_num++;
            }
        }
        
    }
        

    // remove mappoints outliers
    for(auto obj_2d : pKF->mvpLocalObjects_2D)
    {
        //printf("associate obejct with points : %d\n", obj_2d->Obj_c_MapPonits.size());
        // compute the mean and standard.
        obj_2d->ComputeMeanAndStandardFrame();
        // If the object has too few points, ignore.
        if (obj_2d->Obj_c_MapPonits.size() < 8)
            continue;
        // Erase outliers by boxplot.
        obj_2d->RemoveOutliersByBoxPlot(pKF->GetPose());

        // construct feature points rect
        vector<float> x_pt;
        vector<float> y_pt;
        for (auto &pMP : obj_2d->Obj_c_MapPonits)
        {
            float u = pMP->feature.pt.x;
            float v = pMP->feature.pt.y;

            x_pt.push_back(u);
            y_pt.push_back(v);
        }

        if (x_pt.size() < 4) // ignore.
            continue;
        // extremum in xy(uv) direction
        sort(x_pt.begin(), x_pt.end());
        sort(y_pt.begin(), y_pt.end());
        float x_min = x_pt[0];
        float x_max = x_pt[x_pt.size() - 1];
        float y_min = y_pt[0];
        float y_max = y_pt[y_pt.size() - 1];

        // make insure in the image.
        if (x_min < 0)
            x_min = 0;
        if (y_min < 0)
            y_min = 0;
        if (x_max > pKF->mnCols)
            x_max = pKF->mnCols;
        if (y_max > pKF->mnRows)
            y_max = pKF->mnRows;

        // the bounding box constructed by object feature points.
        obj_2d->mRectFeaturePoints = cv::Rect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1);
    }

    // mark bad objects
    pKF->CheckObjectDetection();
}

void Tracking::associateObjectsInFrame(Frame &CurrentFrame){
    // For VOOM
    int N = CurrentFrame.mvpFrameDetections.size();
    // 3D object matches
    CurrentFrame.FrameMapObjects = vector<MapObject*>(N,static_cast<MapObject*>(NULL));
    for (auto det : CurrentFrame.mvpFrameDetections)
    {
        std::string str_flag("EAO");
        int id = det->ObjectDataAssociationFrame(mpMap, &CurrentFrame, str_flag);
        int instance_id = det->instance_id;
        if(id==-1){
            continue;
        }else{
            CurrentFrame.FrameMapObjects[instance_id] = mpMap->mvpMapObjects[id];
        }
        
    }
}

float Tracking::AssociateFeaturesWithObjects(Frame &CurrentFrame){
    // For VOOM
    int N = CurrentFrame.mvpFrameDetections.size();
    // 3D object matches
    FrameMapObjects = vector<MapObject*>(N,static_cast<MapObject*>(NULL));
    for (auto det : CurrentFrame.mvpFrameDetections) 
    {
        int track_id = det->track_id;
        int instance_id = det->instance_id;
        for(auto pMO : mpMap->GetGoodMapObjects())
        {
            if(track_id == pMO->mnId)
            {
                FrameMapObjects[instance_id] = pMO;
                break;
            } 
        }
    }

    float sum_intersection_ratio = 0.0f;
    float count_useful_objects = 0.0f;
    //CurrentFrame.mvpMapPoints = std::vector<MapPoint*>(CurrentFrame.N,static_cast<MapPoint*>(NULL));   
    int count_all_new_mps = 0;
    for(int i = 0; i < FrameMapObjects.size(); i++){
        auto obj = FrameMapObjects[i];
        if(!obj)continue;
        cv::Rect curr_rect = CurrentFrame.mvpFrameDetections[i]->mBoxRect;
        Eigen::Vector4d bb_det; //x_min y_min x_max y_max
        bb_det[0] = curr_rect.x;
        bb_det[1] = curr_rect.y;
        bb_det[2] = curr_rect.x + curr_rect.width;
        bb_det[3] = curr_rect.y + curr_rect.height;;

        auto vIndices_in_box = CurrentFrame.GetFeaturesInBox(bb_det[0], bb_det[2], bb_det[1], bb_det[3]);
        std::vector<MapPoint*> asscociated_mps = obj->GetAssociatedMP();

        if(vIndices_in_box.size() < 3 || asscociated_mps.size() < 3) continue;
        std::set<MapPoint*> set_mp_in_box;
        int original_size_features = vIndices_in_box.size();
        auto iter = vIndices_in_box.begin();
        // 区分keypoints和mappoints
        while(iter != vIndices_in_box.end()){
            MapPoint* mp = CurrentFrame.mvpMapPoints[*iter];
            if(!mp) ++iter;
            else{
                set_mp_in_box.insert(mp);
                iter = vIndices_in_box.erase(iter);
            }
        }
        std::vector<MapPoint*> non_intersected_mps = std::vector<MapPoint*>();
        int count_intersected_mps = 0;
        // 如果object points不属于bbox范围内的mappoints, push进non_intersected_mps
        for(auto mp : obj->GetAssociatedMP()){
            if(set_mp_in_box.count(mp)==0){
                non_intersected_mps.push_back(mp);
                continue;
            }
            else{
                count_intersected_mps += 1;
            }
        }
        
        // 对box中的keypoints与non_intersected_mps进行关联
        //if(intersection_ratio < 0.75f){//ADD MORE ACCOCIATED MAPPOINTS INTO.
        for(auto ind : vIndices_in_box){
            const cv::Mat &dF = CurrentFrame.mDescriptors.row(ind);
            int bestDist = 256;
            int bestIdx = -1;
            for(size_t i=0; i<non_intersected_mps.size(); i++){
                auto pMP = non_intersected_mps[i];
                const cv::Mat dMP = pMP->GetDescriptor();
                const int dist = DescriptorDistance(dMP,dF);
                
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
            if(bestDist<=50)
            {
                //map_obj->InsertNewAsscoatedMapPoint(vpPoints[bestIdx]);
                CurrentFrame.mvpMapPoints[ind] = non_intersected_mps[bestIdx];
                non_intersected_mps.erase(non_intersected_mps.begin()+bestIdx);
                //nmatches_points++;
                count_intersected_mps += 1;
                count_all_new_mps += 1;
            }
        }
        //}
        float intersection_ratio = float(count_intersected_mps)/float(original_size_features);
        count_useful_objects += 1.0f;
        sum_intersection_ratio += intersection_ratio;
    }

    
    //if(count_all_new_mps<20) return 0.0f;

    // Optimize frame pose with all matches
    /*Optimizer::PoseOptimization(&CurrentFrame);

    // Discard outliers
    for(int i =0; i<CurrentFrame.N; i++)
    {
        if(CurrentFrame.mvpMapPoints[i])
        {
            if(CurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = CurrentFrame.mvpMapPoints[i];

                CurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                CurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = CurrentFrame.mnId;
            }
        }
    } */

    return sum_intersection_ratio/(count_useful_objects+0.001f);
}

int Tracking::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void Tracking::AssociateObjets(KeyFrame* pKF)
{
    // step 10.0 associate 2d object with map points
    AssociateObjectMapPoint(pKF);

    // step 10.1 points of the object that appeared in the last 30 frames 
    // are projected into the image to form a projection bounding box.
    for (auto pMO : (mpMap->mvpMapObjects))
    {
        // if (pMO->isBad())
        //     continue;
 
        // object appeared in the last 30 frames.
        if (pMO->GetLatestKeyFrame()->mnFrameId > mCurrentFrame.mnId - 30)
            pMO->ComputeProjectRectFrame(pKF);
        else
        {
           pMO->mRectProject = cv::Rect(0, 0, 0, 0);
        }
    }
    // step 10.2 data association.
    //printf("objecs_kf size: %d \n",  pKF->mvpLocalObjects_2D.size());
    for (auto obj_2d : pKF->mvpLocalObjects_2D)
    {
        if(obj_2d->bad)  continue;

        std::string str_flag("EAO");
        obj_2d->ObjectDataAssociation(mpMap, pKF, str_flag);
    }
    // step 10.3 remove objects with too few observations.
    // maybe can use local map objects scheme
    // for (auto pMO : (mpMap->mvpMapObjects))
    // {
    //     if (pMO->isBad())
    //         continue;

    //     size_t df = pMO->Observations();
    //     if (df < 5)
    //     {
    //         // not been observed in the last 30 frames.
    //         if (pMO->GetLatestKeyFrame()->mnId < (mpLastKeyFrame->mnId - 20))
    //         {
    //             if (df < 3)
    //                 pMO->SetBadFlag();
    //             // if not overlap with other objects, don't remove.
    //             else
    //             {
    //                 bool overlap = false;
    //                 for (auto pMO2 : (mpMap->mvpMapObjects))
    //                 {
    //                     if (pMO2->isBad() || (pMO->mnId == pMO2->mnId))
    //                         continue;

    //                     if (pMO->WhetherOverlap(pMO2))
    //                     {
    //                         overlap = true;
    //                         break;
    //                     }
    //                 }
    //                 if (overlap)
    //                     pMO->SetBadFlag();
    //             }
    //         }
    //     }
    // }

    // step 10.4 Update the co-view relationship between objects. (appears in the same frame).
    for (auto pMO : (mpMap->mvpMapObjects))
    {
        // 最近观察到的object
        if (pMO->GetLatestKeyFrame()->mnFrameId == mCurrentFrame.mnId)
        {
            for (auto pMO2 : (mpMap->mvpMapObjects))
            {
                if (pMO->mnId == pMO2->mnId)
                    continue;

                if (pMO2->GetLatestKeyFrame()->mnFrameId == mCurrentFrame.mnId)
                {
                    int nObjId = pMO2->mnId;

                    map<int, int>::iterator sit;
                    sit = pMO->mmAppearSametime.find(nObjId);

                    if (sit != pMO->mmAppearSametime.end())
                    {
                        int sit_sec = sit->second;
                        pMO->mmAppearSametime.erase(nObjId);
                        pMO->mmAppearSametime.insert(make_pair(nObjId, sit_sec + 1));
                    }
                    else
                        pMO->mmAppearSametime.insert(make_pair(nObjId, 1));   // first co-view.
                }
            }
        }
    }
    // step 10.5 Merge potential associate objects (see mapping thread).

    // update map quadric
    InitializeQuadric(pKF);
    
}

void Tracking::InitializeQuadric(KeyFrame* pKF)
{
    // update quadric solver
    if(!mpQuadricSolver)
    {
        mpQuadricSolver = new QuadricSolver(pKF->mnRows, pKF->mnCols, mK);
    }
    // 遍历Map的MapObjects
    for (auto pMO : mpMap->mvpMapObjects)
    {
        //  update current iou
        pMO->UpdateCurrentIou();

        //  extract cuboid
        pMO->ExtractCuboid();

        // check all abnormal initialized objects
        if(pMO->mbIsInitialized && !pMO->CheckSelfQuality(pKF))
        {
            pMO->mbIsInitialized = false;
            printf("bad object: %d ,re-initialization!\n",pMO->mnId);
        }

        std::map<KeyFrame*, size_t> obs = pMO->GetObservations();
        // 如观测少于3帧，跳过该MapObject的初始化
        if(obs.size() < 3) 
            continue;
        Vector9d vec_paras;

        vec_paras = mpQuadricSolver->initializeQuadricConvexHull(obs);

        if(mpQuadricSolver->isGood())
        {
            pMO->mpBoxQuadric->fromMinimalVector(vec_paras);
            if(pMO->CheckQuality(pKF,pMO->mpBoxQuadric))
            {
                pMO->mBoxDistance = pMO->GetMeanTangentDistance(pMO->mpBoxQuadric,mK);
                
                pMO->mBoxIou = pMO->GetMeanConicPolygonIoU(pMO->mpBoxQuadric,mK,false);
                
                // whether update Quadric
                if(pMO->mBoxDistance < pMO->mCurrentDistance && pMO->mBoxIou > 0.4 && pMO->mBoxDistance != -1)
                {
                    pMO->UpdateQuadric(*(pMO->mpBoxQuadric));
                    pMO->mCurrentDistance = pMO->mBoxDistance;
                    pMO->mCurrentIou = pMO->mBoxIou;
                    pMO->mbIsInitialized = true;
                }
                
            }
        }
    }
}

void Tracking::DrawQuadricProject(cv::Mat &im, bool draw_local_objects)
{   
    // draw projection
    int nLineWidth = 1;
    int nLatitudeNum = 6;
    int nLongitudeNum = 5;

    std::vector<MapObject*> vpMOs;
    if(draw_local_objects)
    {
        vpMOs = mvpLocalMapObjects;
    }
    else
    {
        vpMOs = mpMap->GetGoodMapObjects();
    }

    for (auto pMO:vpMOs)
    {
        // step 10.7 project quadrics to the image (only for visualization).
        cv::Mat axe = cv::Mat::zeros(3, 1, CV_32F);
        axe.at<float>(0) = pMO->mpQuadric->scale[0];
        axe.at<float>(1) = pMO->mpQuadric->scale[1];
        axe.at<float>(2) = pMO->mpQuadric->scale[2];

        // object pose (world).
        cv::Mat Twq = Converter::toCvMat(pMO->mpQuadric->pose);

        // Projection Matrix K[R|t].
        cv::Mat P(3, 4, CV_32F);
        cv::Mat Tcw = mCurrentFrame.mTcw;
        P = Tcw.rowRange(0,3).colRange(0,4).clone();
        P = mCurrentFrame.mK * P;

        // draw params
        cv::Scalar sc = paras.color_vec[pMO->mnClassId % 7];

        // generate angluar grid -> xyz grid (vertical half sphere)
        vector<float> vfAngularLatitude;  // (-90, 90)
        vector<float> vfAngularLongitude; // [0, 180]
        cv::Mat pointGrid(nLatitudeNum + 2, nLongitudeNum + 1, CV_32FC4);

        for (int i = 0; i < nLatitudeNum + 2; i++)
        {
            float fThetaLatitude = -M_PI_2 + i * M_PI / (nLatitudeNum + 1);
            cv::Vec4f *p = pointGrid.ptr<cv::Vec4f>(i);
            for (int j = 0; j < nLongitudeNum + 1; j++)
            {
                float fThetaLongitude = j * M_PI / nLongitudeNum;
                p[j][0] = axe.at<float>(0, 0) * cos(fThetaLatitude) * cos(fThetaLongitude);
                p[j][1] = axe.at<float>(1, 0) * cos(fThetaLatitude) * sin(fThetaLongitude);
                p[j][2] = axe.at<float>(2, 0) * sin(fThetaLatitude);
                p[j][3] = 1.;
            }
        }

        // draw latitude
        for (int i = 0; i < pointGrid.rows; i++)
        {
            cv::Vec4f *p = pointGrid.ptr<cv::Vec4f>(i);
            // [0, 180]
            for (int j = 0; j < pointGrid.cols - 1; j++)
            {
                cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << p[j][0], p[j][1], p[j][2], p[j][3]);
                cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << p[j + 1][0], p[j + 1][1], p[j + 1][2], p[j + 1][3]);
                cv::Mat conicPt0 = P * Twq * spherePt0;
                cv::Mat conicPt1 = P * Twq * spherePt1;
                cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
                cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
                cv::line(im, pt0, pt1, sc, nLineWidth); // [0, 180]
            }
            // [180, 360]
            for (int j = 0; j < pointGrid.cols - 1; j++)
            {
                cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << -p[j][0], -p[j][1], p[j][2], p[j][3]);
                cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << -p[j + 1][0], -p[j + 1][1], p[j + 1][2], p[j + 1][3]);
                cv::Mat conicPt0 = P * Twq * spherePt0;
                cv::Mat conicPt1 = P * Twq * spherePt1;
                cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
                cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
                cv::line(im, pt0, pt1, sc, nLineWidth); // [180, 360]
            }
        }

        // draw longitude
        cv::Mat pointGrid_t = pointGrid.t();
        for (int i = 0; i < pointGrid_t.rows; i++)
        {
            cv::Vec4f *p = pointGrid_t.ptr<cv::Vec4f>(i);
            // [0, 180]
            for (int j = 0; j < pointGrid_t.cols - 1; j++)
            {
                cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << p[j][0], p[j][1], p[j][2], p[j][3]);
                cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << p[j + 1][0], p[j + 1][1], p[j + 1][2], p[j + 1][3]);
                cv::Mat conicPt0 = P * Twq * spherePt0;
                cv::Mat conicPt1 = P * Twq * spherePt1;
                cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
                cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
                cv::line(im, pt0, pt1, sc, nLineWidth); // [0, 180]
            }
            // [180, 360]
            for (int j = 0; j < pointGrid_t.cols - 1; j++)
            {
                cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << -p[j][0], -p[j][1], p[j][2], p[j][3]);
                cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << -p[j + 1][0], -p[j + 1][1], p[j + 1][2], p[j + 1][3]);
                cv::Mat conicPt0 = P * Twq * spherePt0;
                cv::Mat conicPt1 = P * Twq * spherePt1;
                cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
                cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
                cv::line(im, pt0, pt1, sc, nLineWidth); // [180, 360]
            }
        }

        // draw projection bbox
        g2o::SE3Quat cam_pose = Converter::toSE3Quat(Tcw);
        Eigen::Matrix3d Kalib = Converter::toMatrix3d(mK);

        Vector4d bbox = pMO->mpQuadric->getBoundingBoxFromDualEllipse(cam_pose, Kalib);
        cv::Rect rect(bbox(0), bbox(1), bbox(2)-bbox(0)+1, bbox(3)-bbox(1)+1 );
        // cv::rectangle(im, rect, sc, 1);
        
        // draw label text
        // std::string label = paras.coco_class_name_map[pMO->mnClassId];
        std::string label = paras.coco_class_name_map[pMO->mnClassId] + "_" + std::to_string(pMO->mnId);
        //std::string label = std::to_string(pMO->mnId);
        cv::Size label_size = cv::getTextSize(label,cv::FONT_HERSHEY_SIMPLEX,0.7,1.6,0);
        cv::Rect rect_text(rect.x, rect.y-label_size.height, label_size.width, label_size.height);

        cv::Mat mask_img = cv::Mat::zeros(im.size(),im.type());
        cv::rectangle(mask_img,rect_text,sc,-1);
        cv::addWeighted(mask_img,0.7,im,1,0,im);
        // draw bounding box.
        //cv::rectangle(im, rect, sc,2);
        // draw text
        cv::putText(im, label,cv::Point(rect.x, rect.y-1), cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(0,0,0),1.6);
    }
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->SetReferenceMapObjects(mvpLocalMapObjects);
    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
    UpdateLocalObjects();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalObjects()
{
    mvpLocalMapObjects.clear();
    std::set<MapObject*> spMOs; //temp container to avoid duplication
    std::vector<KeyFrame*> localKFs = mpLastKeyFrame->GetBestCovisibilityKeyFrames(30);
    for(auto pKF : localKFs)
    {
        // get filtered map objects
        const vector<MapObject*> vpMOs = pKF->GetMapObjects();
        spMOs.insert(vpMOs.begin(),vpMOs.end());
    }

    // add objects observed by points of current frame
    std::vector<MapPoint*> vpMPs = mCurrentFrame.GetFrameMapPoints();
    for(auto pMP:vpMPs)
    {
        if(pMP->mpLastObservedObject)
        {
            if(pMP->mpLastObservedObject->mbIsInitialized)
                spMOs.insert(pMP->mpLastObservedObject);
        }    
    }

    for(auto pMO:spMOs)
    {
        mvpLocalMapObjects.push_back(pMO);
    }
    spMOs.clear();
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    //Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} //namespace ORB_SLAM
