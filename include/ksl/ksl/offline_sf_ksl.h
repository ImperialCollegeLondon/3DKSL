#ifndef OFFLINE_SF_KSL_H
#define OFFLINE_SF_KSL_H

#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <ksl/affinity/sparse_subspace.h>
#include <ksl/clustering/spectral_clustering.h>
#include <ksl/flow/pd_scene_flow.h>
#include <ksl/utils/pcloud.h>
#include <opencv2/opencv.hpp>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace ksl
{

class OfflineKinematicStructure
{

private:

  utils::CamParams<float> camParams_;
  const int rows_;

protected:

  const int frames_;
  int frame_;
  const int nMaxClusters_;
  int nPoints_;
  float resolution_;
  const float rho_, eigThresh_;

  Eigen::VectorXi lostPointsInd_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud_, pcloudSubSampled_;

  Eigen::MatrixXf dataPoints_;

  flow::PDSceneFlow sceneFlow_;

  affinity::SparseSubspace<float> a_;
  clustering::SpectralClustering<float> c_;

public:

  OfflineKinematicStructure(
    const int& width, const int& height,
    const int& rows,
    const int& frames,
    const int& nMaxClusters,
    const int& nPoints=500,
    const float& rho=1.0,
    const float& eigThresh=0.5);
  OfflineKinematicStructure(
    const int& width, const int& height,
    const int& rows,
    const int& frames,
    const int& nMaxClusters,
    const float& resolution,
    const float& rho=1.0,
    const float& eigThresh=0.5);
  ~OfflineKinematicStructure(void);

  inline const int&
  width(void) const
  {
    return camParams_.width;
  }
  inline const int&
  height(void) const
  {
    return camParams_.height;
  }
  inline const float&
  cx(void) const
  {
    return camParams_.cx;
  }
  inline const float&
  cy(void) const
  {
    return camParams_.cy;
  }
  inline const float&
  fx(void) const
  {
    return camParams_.fx;
  }
  inline const float&
  fy(void) const
  {
    return camParams_.fy;
  }
  inline const int&
  frames(void) const
  {
    return frames_;
  }
  inline const int&
  nMaxClusters(void) const
  {
    return nMaxClusters_;
  }
  inline const int&
  nPoints(void) const
  {
    return nPoints_;
  }
  inline const float&
  resolution(void) const
  {
    return resolution_;
  }
  inline const float&
  rho(void) const
  {
    return rho_;
  }
  inline const float&
  eigThresh(void) const
  {
    return eigThresh_;
  }
  inline const pcl::PointCloud<pcl::PointXYZ>::ConstPtr
  pcloud(void) const
  {
    return pcloud_;
  }
  inline const pcl::PointCloud<pcl::PointXYZ>::ConstPtr
  pcloudSubSampled(void) const
  {
    return pcloudSubSampled_;
  }
  inline const int&
  nGroups(void) const
  {
    return c_.nClusters();
  }
  inline const Eigen::VectorXi&
  groups(void) const
  {
    return c_.clusters();
  }

  void
  computeKSL(void);

  void
  computeSceneFlow(
    const cv::Mat& img1, const cv::Mat& img2,
    const cv::Mat& depth1, const cv::Mat& depth2);

  void
  trackPoints(
    const cv::Mat& img,
    const cv::Mat& depth,
    const float& dEps=5.0e-3);

protected:

  void
  mapPoints(void);

  void
  subSamplePCloud(void);

private:

};

}

#endif // OFFLINE_SF_KSL_H
