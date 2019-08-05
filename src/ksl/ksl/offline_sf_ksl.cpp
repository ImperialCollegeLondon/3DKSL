#include <ksl/ksl/offline_sf_ksl.h>

namespace ksl
{
  
OfflineKinematicStructure::OfflineKinematicStructure(
  const int& width, const int& height,
  const int& rows,
  const int& frames,
  const int& nMaxClusters,
  const int& nPoints,
  const float& rho,
  const float& eigThresh):
  camParams_(width, height),
  rows_(rows),
  frames_(frames), frame_(0),
  nMaxClusters_(nMaxClusters),
  nPoints_(nPoints),
  resolution_(0.0),
  rho_(rho), eigThresh_(eigThresh),
  pcloud_(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(camParams_.width, camParams_.height))),
  pcloudSubSampled_(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>)),
  sceneFlow_(camParams_.width, camParams_.height, rows_),
  a_(0.0, rho_, 0, 0, 200, true, false, 5.0e-4, 5.0e-4),
  c_(eigThresh_, 1.0e-3, 1.0e-6, nMaxClusters_, 200)
{
  assert(camParams_.width>0 && camParams_.height>0 && nPoints_>0);
}

OfflineKinematicStructure::OfflineKinematicStructure(
  const int& width, const int& height,
  const int& rows,
  const int& frames,
  const int& nMaxClusters,
  const float& resolution,
  const float& rho,
  const float& eigThresh):
  camParams_(width, height),
  rows_(rows),
  frames_(frames), frame_(0),
  nMaxClusters_(nMaxClusters),
  nPoints_(0),
  resolution_(resolution),
  rho_(rho), eigThresh_(eigThresh),
  pcloud_(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(camParams_.width, camParams_.height))),
  pcloudSubSampled_(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>)),
  sceneFlow_(camParams_.width, camParams_.height, rows_),
  a_(0.0, rho_, 0, 0, 200, true, false, 5.0e-4, 5.0e-4),
  c_(eigThresh_, 1.0e-3, 1.0e-6, nMaxClusters_, 200)
{
  assert(camParams_.width>0 && camParams_.height>0 && resolution_>0.0);
}

OfflineKinematicStructure::~OfflineKinematicStructure(void)
{}

void
OfflineKinematicStructure::computeKSL(void)
{
  int nLostPoints=0;
  for(int k=0; k<nPoints_; ++k)
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
      ++nLostPoints;
    }
  }
  /* assumed conservative resize, because the new size is always leq than the previous */
  const int nPointsSubSampled=nPoints_-nLostPoints;
  pcloudSubSampled_->resize(nPointsSubSampled);
  dataPoints_.conservativeResize(Eigen::NoChange, nPointsSubSampled);
  c_.compute(dataPoints_.transpose(), &a_);
}

void
OfflineKinematicStructure::computeSceneFlow(
  const cv::Mat &img1, const cv::Mat &img2,
  const cv::Mat &depth1, const cv::Mat &depth2)
{
  assert(img1.cols==camParams_.width && depth1.cols==camParams_.width);
  assert(img1.rows==camParams_.height && depth1.rows==camParams_.height);
  assert(img2.cols==camParams_.width && depth2.cols==camParams_.width);
  assert(img2.rows==camParams_.height && depth2.rows==camParams_.height);

  cv::Mat gray1, gray2;
  cv::cvtColor(img1, gray1, cv::COLOR_RGB2GRAY);
  cv::cvtColor(img2, gray2, cv::COLOR_RGB2GRAY);
  sceneFlow_.compute(gray1, gray2, depth1, depth2);

   /* first frame */
  if(frame_==0)
  {
    utils::depthToPCloud<float>(img1, depth1, camParams_, pcloud_);
    utils::fastBilateralFilter<float, pcl::PointXYZ>(5.0, 5.0e-3, pcloud_);
    subSamplePCloud();
    lostPointsInd_.setZero(nPoints_);
    dataPoints_.resize(3*frames_, nPoints_);
    mapPoints();
  }
  ++frame_;
}

void
OfflineKinematicStructure::trackPoints(
  const cv::Mat& img,
  const cv::Mat& depth,
  const float& dEps)
{
  utils::depthToPCloud<float>(img, depth, camParams_, pcloud_);
  //utils::fastBilateralFilter<float, pcl::PointXYZ>(5.0, 5.0e-3, pcloud_);

  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution_);
  octree.setInputCloud(pcloud_);
  octree.addPointsFromInputCloud();

  std::vector<int> indVec(1);
  std::vector<float> distVec(1);

  /* mapping interface */
  Eigen::Map<const Eigen::VectorXf> dxp(sceneFlow_.dxp(), sceneFlow_.size());
  Eigen::Map<const Eigen::VectorXf> dyp(sceneFlow_.dyp(), sceneFlow_.size());
  Eigen::Map<const Eigen::VectorXf> dzp(sceneFlow_.dzp(), sceneFlow_.size());

  int i, j;
  for(int k=0; k<nPoints_; ++k)
  {
    utils::xyzToIJ<float>(
      pcloudSubSampled_->at(k).x, pcloudSubSampled_->at(k).y, pcloudSubSampled_->at(k).z,
      camParams_, &i, &j);
    if(!(0<=i && i<camParams_.height && 0<=j && j<camParams_.width))
    {
      ++lostPointsInd_(k);
      continue;
    }
    int ind=((i+j*rows_)*rows_)/camParams_.height;
    if(!(0<=ind && ind<sceneFlow_.size()))
    {
      ++lostPointsInd_(k);
      continue;
    }

    pcloudSubSampled_->at(k).x+=dyp(ind);
    pcloudSubSampled_->at(k).y-=dzp(ind);
    pcloudSubSampled_->at(k).z+=dxp(ind);

    octree.nearestKSearch(pcloudSubSampled_->at(k), 1, indVec, distVec);
    if(pcloud_->at(indVec[0]).z>0.0)
    {
      if(std::abs(pcloud_->at(indVec[0]).x-pcloudSubSampled_->at(k).x)<dEps &&
        std::abs(pcloud_->at(indVec[0]).y-pcloudSubSampled_->at(k).y)<dEps &&
        std::abs(pcloud_->at(indVec[0]).z-pcloudSubSampled_->at(k).z)<dEps)
      {
        pcloudSubSampled_->at(k)=pcloud_->at(indVec[0]);
      }
      else
      {
        ++lostPointsInd_(k);
      }
    }
    else
    {
      ++lostPointsInd_(k);
    }
  }
  mapPoints();
}

void
OfflineKinematicStructure::mapPoints(void)
{
  dataPoints_.middleRows(3*frame_, 3)=pcloudSubSampled_->getMatrixXfMap(3, 4, 0);
}

void
OfflineKinematicStructure::subSamplePCloud(void)
{
  if(nPoints_>0)
  {
    utils::subSamplePCloud<float, pcl::PointXYZ>(pcloud_, nPoints_,
      pcloudSubSampled_);
    resolution_=5.0e-3;
  }
  else
  {
    utils::subSamplePCloud<float, pcl::PointXYZ>(pcloud_, resolution_,
      pcloudSubSampled_);
    nPoints_=pcloudSubSampled_->size();
  }
}

}
