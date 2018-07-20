#include <ksl/offline_ksl/offline_sf_ksl.hpp>
#include <iostream>

namespace ksl
{

namespace offline_ksl
{

template<typename pointT>
KinematicStructureLearning<pointT>::KinematicStructureLearning(
  const int width, const int height,
  const int rows,
  const int frames,
  const float cx, const float cy,
  const int outlier,
  const float rho,
  const float fx, const float fy):
  width_(width), height_(height),
  rows_(rows),
  frames_(frames), frame_(0),
  cx_(cx), cy_(cy),
  outlier_(outlier),
  rho_(rho),
  fx_(fx), fy_(fy),
  ratioHeightRows_((float) height_/rows_),
  pcloud_(typename pcl::PointCloud<pointT>::Ptr(new typename pcl::PointCloud<pointT>(width_, height_))),
  pcloudSubSampled_(typename pcl::PointCloud<pointT>::Ptr(new typename  pcl::PointCloud<pointT>)),
  sceneFlow_(width_, height_, rows_)
{
  assert(frames_>0);
}

template<typename pointT>
KinematicStructureLearning<pointT>::~KinematicStructureLearning(void)
{}

template<typename pointT> float
KinematicStructureLearning<pointT>::compute(
  const cv::Mat &rgb1,
  const cv::Mat &rgb2,
  const cv::Mat &depth1,
  const cv::Mat &depth2,
  const float nPointsSubSampled)
{
  assert(rgb1.cols==width_ && depth1.cols==width_ && rgb1.rows==height_ && depth1.rows==height_);
  assert(rgb2.cols==width_ && depth2.cols==width_ && rgb2.rows==height_ && depth2.rows==height_);

  frame_++;
  //std::cout<<"frame: "<<frame_<<std::endl;

  cv::Mat gray1, gray2;
  cv::cvtColor(rgb1, gray1, cv::COLOR_RGB2GRAY);
  cv::cvtColor(rgb2, gray2, cv::COLOR_RGB2GRAY);
  sceneFlow_.compute(gray1, gray2, depth1, depth2);
  /* mapping interface */
  Eigen::Map<const Eigen::VectorXf> dxp(sceneFlow_.dxp(), sceneFlow_.size());
  Eigen::Map<const Eigen::VectorXf> dyp(sceneFlow_.dyp(), sceneFlow_.size());
  Eigen::Map<const Eigen::VectorXf> dzp(sceneFlow_.dzp(), sceneFlow_.size());

  if(frame_==1)
  {
    depthToPCloud(rgb1, depth1);
    fastBilateralFilter();

    if(nPointsSubSampled>1)
    {
      subSamplePCloud(nPointsSubSampled);
    }
    else if(nPointsSubSampled!=0.0)
    {
      subSamplePCloud(Eigen::Vector4f(nPointsSubSampled, nPointsSubSampled, nPointsSubSampled, 0.0));
    }
    else
    {
      subSamplePCloud();
    }
    nPointsSubSampled_=pcloudSubSampled_->size();

    lostPointsInd_=Eigen::VectorXi::Zero(nPointsSubSampled_);
    resizeDataPoints();
    mapDataPoints();

    frame_++;
  }
  if(frame_<=frames_)
  {
    depthToPCloud(rgb2, depth2);
    //fastBilateralFilter();

    float dEps=5e-3;
    pcl::octree::OctreePointCloudSearch<pointT> octree(dEps);
    octree.setInputCloud(pcloud_);
    octree.addPointsFromInputCloud();
    std::vector<int> indVec(1);
    std::vector<float> indVecDist(1);

    int i, j, ind;
    for(int k=0; k<nPointsSubSampled_; k++)
    {
      xyzToIJ(pcloudSubSampled_->at(k).x,
        pcloudSubSampled_->at(k).y, pcloudSubSampled_->at(k).z, &j, &i);
      if(!(i>=0 && j>=0 && i<height_ && j<width_))
      {
        lostPointsInd_(k)++;
        continue;
      }
      ind=(i+j*rows_)/ratioHeightRows_;
      if(!(ind>=0 && ind<sceneFlow_.size()))
      {
        lostPointsInd_(k)++;
        continue;
      }

      //float x=pcloudSubSampled_->at(k).x;
      //float y=pcloudSubSampled_->at(k).y;
      //float z=pcloudSubSampled_->at(k).z;

      pcloudSubSampled_->at(k).x+=dyp(ind);
      pcloudSubSampled_->at(k).y-=dzp(ind);
      pcloudSubSampled_->at(k).z+=dxp(ind);

      octree.nearestKSearch(pcloudSubSampled_->at(k), 1, indVec, indVecDist);
      if(pcloud_->at(indVec[0]).z>0.0)
      {
        if(fabs(pcloud_->at(indVec[0]).x-pcloudSubSampled_->at(k).x)<dEps &&
          fabs(pcloud_->at(indVec[0]).y-pcloudSubSampled_->at(k).y)<dEps &&
          fabs(pcloud_->at(indVec[0]).z-pcloudSubSampled_->at(k).z)<dEps)
        {
          pcloudSubSampled_->at(k)=pcloud_->at(indVec[0]);
        }
        else
        {
          lostPointsInd_(k)++;
        }
      }
      else
      {
        lostPointsInd_(k)++;
      }
    }
    mapDataPoints();
  }
  if(frame_>=frames_)
  {
    int nLostPoints=0;
    for(int k=0; k<nPointsSubSampled_; k++)
    {
      if(lostPointsInd_(k)<1)
      {
        if(nLostPoints>0)
        {
          pcloudSubSampled_->at(k-nLostPoints)=pcloudSubSampled_->at(k);
          dataPoints_.col(k-nLostPoints)=dataPoints_.col(k);
        }
      }
      else
      {
        nLostPoints++;
      }
    }
    /* assumed conservative resize, because the new size is always leq than the previous */
    nPointsSubSampled_-=nLostPoints;
    pcloudSubSampled_->resize(nPointsSubSampled_);
    dataPoints_.conservativeResize(Eigen::NoChange, nPointsSubSampled_);
    std::cout<<"number of effective sub-sampled points: "<<nPointsSubSampled_<<"; ";
    std::cout<<"number of lost points: "<<nLostPoints<<std::endl;
    clock_t t=clock();
    subspaceClustering_.compute(dataPoints_, 0.0, 0, 
      (frames_>1) ? 1 : 0, outlier_, 0, rho_, 0, ceilf(log2f(nPointsSubSampled_)), 0.1);
    t=clock()-t;
    std::cout<<"total time elapsed: "<<((float) t/CLOCKS_PER_SEC)<<" [sec]"<<std::endl;
    frame_=0;
    return ((float) t/CLOCKS_PER_SEC);
  }
  return 0.0;
}

template<typename pointT> void
KinematicStructureLearning<pointT>::fastBilateralFilter(
  const float sigmaS,
  const float sigmaR)
{
  typename pcl::FastBilateralFilterOMP<pointT> fbf;
  fbf.setSigmaS(sigmaS);
  fbf.setSigmaR(sigmaR);
  fbf.setInputCloud(pcloud_);
  fbf.applyFilter(*pcloud_);
}

template<typename pointT> void
KinematicStructureLearning<pointT>::subSamplePCloud(
  const Eigen::Vector4f &subSamplingLeafSize)
{
  typename pcl::VoxelGrid<pointT> subSamplingFilter;
  subSamplingFilter.setLeafSize(subSamplingLeafSize);
  subSamplingFilter.setInputCloud(pcloud_);
  subSamplingFilter.filter(*pcloudSubSampled_);
}

template<typename pointT> void
KinematicStructureLearning<pointT>::subSamplePCloud(
  const int nSample)
{
  std::vector<int> ind;
  pcl::removeNaNFromPointCloud(*pcloud_, *pcloudSubSampled_, ind);
  typename pcl::RandomSample<pointT> subSamplingFilter;
  subSamplingFilter.setSample(nSample);
  subSamplingFilter.setInputCloud(pcloudSubSampled_);
  subSamplingFilter.filter(*pcloudSubSampled_);
}

template<typename pointT> void
KinematicStructureLearning<pointT>::ijToXYZ(
  const int i, const int j,
  const float depth,
  float *x, float *y, float *z)
{
  if(depth>0.0)
  {
    *z=depth;
    *x=*z*(i-cx_)/fx_;
    *y=*z*(cy_-j)/fy_;
  }
  else
  {
    *x=std::numeric_limits<float>::quiet_NaN();
    *y=std::numeric_limits<float>::quiet_NaN();
    *z=std::numeric_limits<float>::quiet_NaN();
  }
}

template<typename pointT> void
KinematicStructureLearning<pointT>::xyzToIJ(
  const float x, const float y, const float z,
  int *i, int *j)
{
  if(z>0.0)
  {
    *i=roundf(cx_+(fx_*x/z));
    *j=roundf(cy_-(fy_*y/z));
  }
  else
  {
    *i=-1;
    *j=-1;
  }
}

KinematicStructureLearningXYZ::KinematicStructureLearningXYZ(
  const int width, const int height,
  const int rows,
  const int frames):
  KinematicStructureLearning(
    width, height,
    rows,
    frames,
    width*0.5, height*0.5)
{}

KinematicStructureLearningXYZ::KinematicStructureLearningXYZ(
  const int width, const int height,
  const int rows,
  const int frames,
  const float cx, const float cy,
  const int outlier,
  const float rho,
  const float fx, const float fy):
  KinematicStructureLearning(
    width, height,
    rows,
    frames,
    cx, cy,
    outlier,
    rho,
    fx, fy)
{}

KinematicStructureLearningXYZ::~KinematicStructureLearningXYZ(void)
{}

void
KinematicStructureLearningXYZ::depthToPCloud(
  const cv::Mat &img,
  const cv::Mat &depth)
{
  pcl::PointXYZ pt;
  pcloud_->is_dense=false;
  for(int i=0; i<height_; i++)
  {
    for(int j=0; j<width_; j++)
    {
      ijToXYZ(j, i, depth.at<float>(i, j), &pt.x, &pt.y, &pt.z);
      pcloud_->at(j, height_-i-1)=pt;
    }
  }
}

void
KinematicStructureLearningXYZ::mapDataPoints(void)
{
  dataPoints_.middleRows(3*(frame_-1), 3)=pcloudSubSampled_->getMatrixXfMap(3, 4, 0);
}

void
KinematicStructureLearningXYZ::resizeDataPoints(void)
{
  dataPoints_.resize(3*frames_, nPointsSubSampled_);
}

KinematicStructureLearningXYZRGB::KinematicStructureLearningXYZRGB(
  const int width, const int height,
  const int rows,
  const int frames):
  KinematicStructureLearning(
    width, height,
    rows,
    frames,
    width*0.5, height*0.5)
{}

KinematicStructureLearningXYZRGB::KinematicStructureLearningXYZRGB(
  const int width, const int height,
  const int rows,
  const int frames,
  const float cx, const float cy,
  const int outlier,
  const float rho,
  const float fx, const float fy):
  KinematicStructureLearning(
    width, height,
    rows,
    frames,
    cx, cy,
    outlier,
    rho,
    fx, fy)
{}

KinematicStructureLearningXYZRGB::~KinematicStructureLearningXYZRGB(void)
{}

void
KinematicStructureLearningXYZRGB::depthToPCloud(
  const cv::Mat &img,
  const cv::Mat &depth)
{
  pcl::PointXYZRGB pt;
  cv::Vec3b pixel;
  pcloud_->is_dense=false;
  for(int i=0; i<height_; i++)
  {
    for(int j=0; j<width_; j++)
    {
      ijToXYZ(j, i, depth.at<float>(i, j), &pt.x, &pt.y, &pt.z);
      pixel=img.at<cv::Vec3b>(i, j);
      pt.b=pixel.val[0];
      pt.g=pixel.val[1];
      pt.r=pixel.val[2];
      pcloud_->at(j, height_-i-1)=pt;
    }
  }
}

void
KinematicStructureLearningXYZRGB::mapDataPoints(void)
{
  dataPoints_.middleRows(4*(frame_-1), 3)=pcloudSubSampled_->getMatrixXfMap(3, 8, 0);
  for(int k=0; k<nPointsSubSampled_; k++)
  {
    dataPoints_(4*(frame_-1)+3, k)=0.001172549*pcloudSubSampled_->at(k).r+
      0.002301961*pcloudSubSampled_->at(k).g+0.000447059*pcloudSubSampled_->at(k).b;
  }
}

void
KinematicStructureLearningXYZRGB::resizeDataPoints(void)
{
  dataPoints_.resize(4*frames_, nPointsSubSampled_);
}

}

}

