#ifndef ONLINE_SF_KSL_H
#define ONLINE_SF_KSL_H

#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <igraph.h>
#include <ksl/affinity/nonmetric_twoviews.h>
#include <ksl/clustering/label_propagation.h>
//#include <ksl/clustering/spectral_clustering.h>
#include <ksl/flow/pd_scene_flow.h>
#include <ksl/utils/dataset.h>
#include <ksl/utils/pcloud.h>
//#include <ksl/utils/tictoc.h>
#include <opencv2/opencv.hpp>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace ksl
{

class OnlineKinematicStructure
{

private:

  //utils::TicToc t_;
  //double tTrackingPoints_, tDist_, tAffinity_, tCluster_, tKS_;

  bool indToogle_;
  const float scale_;
  utils::CamParams<float> camParams_;
  const int rows_;

  Eigen::VectorXf aMean_;
  Eigen::MatrixXf meanDist_, m2Dist_;
  //Eigen::MatrixXf c1Mtx_, c2Mtx_;

protected:

  unsigned int frame_;
  int nPoints_, nLostPoints_, nMinPoints_;
  float resolution_, beta_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcloudOriginal_, pcloud_, pcloudSubSampled_;

  Eigen::VectorXi accTrackPoints_;
  Eigen::MatrixXf dDist_;
  Eigen::Map<const Eigen::MatrixXf> pointsMap_;
  std::vector<Eigen::MatrixXf> pointsPairwiseDistanceEuclidean_;

  flow::PDSceneFlow sceneFlow_;

  affinity::NonMetricTwoViews<float> a_;
  clustering::LabelPropagation<float> c_;
  //clustering::SpectralClustering<float> c_;

  Eigen::MatrixXf ylMtx_;
  Eigen::MatrixXf& yMtx_;

  Eigen::MatrixXf centroids_, gMtx_;
  igraph_t ksGraph_;
  Eigen::MatrixXi ks_;

public:

  OnlineKinematicStructure(
    const int& width, const int& height,
    const float& scale,
    const utils::CamParams<float>& camParams,
    const float& beta,
    const affinity::NonMetricTwoViewsParams<float>& aParams,
    const clustering::LabelPropagationParams<float>& cParams);
  OnlineKinematicStructure(
    const int& width, const int& height,
    const float& scale,
    const utils::CamParams<float>& camParams,
    const float& beta,
    const affinity::NonMetricTwoViewsParams<float>& aParams,
    const clustering::LabelPropagationParams<float>& cParams,
    const int& nPoints);
  OnlineKinematicStructure(
    const int& width, const int& height,
    const float& scale,
    const utils::CamParams<float>& camParams,
    const float& beta,
    const affinity::NonMetricTwoViewsParams<float>& aParams,
    const clustering::LabelPropagationParams<float>& cParams,
    const float& resolution);
  ~OnlineKinematicStructure(void);

  /*inline const double&
  tTrackingPoints(void) const
  {
    return tTrackingPoints_;
  }
  inline const double&
  tDist(void) const
  {
    return tDist_;
  }
  inline const double&
  tAffinity(void) const
  {
    return tAffinity_;
  }
  inline const double&
  tCluster(void) const
  {
    return tCluster_;
  }
  inline const double&
  tKS(void) const
  {
    return tKS_;
  }*/

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
  nPoints(void) const
  {
    return nPoints_;
  }
  inline const int&
  nLostPoints(void) const
  {
    return nLostPoints_;
  }
  inline const int&
  nMinPoints(void) const
  {
    return nMinPoints_;
  }
  inline const float&
  resolution(void) const
  {
    return resolution_;
  }
  inline const float&
  beta(void) const
  {
    return beta_;
  }
  inline const pcl::PointCloud<pcl::PointXYZ>::ConstPtr
  pcloudOriginal(void) const
  {
    return pcloudOriginal_;
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
  inline const Eigen::VectorXi&
  accTrackPoints(void) const
  {
    return accTrackPoints_;
  }
  inline const Eigen::MatrixXf&
  dDist(void) const
  {
    return dDist_;
  }
  inline const Eigen::MatrixXf&
  accAdjacency(void) const
  {
    return a_.wMtx();
  }
  inline const Eigen::MatrixXf&
  pointsPairwiseDistanceEuclidean(void) const
  {
    return pointsPairwiseDistanceEuclidean_[indToogle_];
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
  inline const Eigen::MatrixXf&
  centroids(void) const
  {
    return centroids_;
  }
  inline const int
  nKs(void) const
  {
    return ks_.rows();
  }
  inline const Eigen::MatrixXi&
  ks(void) const
  {
    return ks_;
  }

  void
  computeKSL(
    const int& opt,
    const int& minEvidence=15);
  void
  computeSceneFlow(
    const cv::Mat& img1, const cv::Mat& img2,
    const cv::Mat& depth1, const cv::Mat& depth2);

  void
  trackPoints(
    const cv::Mat& img,
    const cv::Mat& depth,
    const int& minEvidence=15,
    const float& dEps=5.0e-3);

protected:

  void
  computePointsPairwiseDistanceEuclidean(void);

  inline void
  initKSGraph(
    const int& nVertices=0);
  inline void
  createKSGraph(void);
  inline void
  destroyKSGraph(void);

  inline void
  mapPoints(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pcloud);

  inline void
  subSamplePCloud(void);
  inline void
  subSamplePCloud(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& pcloudSubSampled);

private:

  void
  computeDist(
    const int& opt,
    const int& minEvidence);

};

}

#endif // ONLINE_SF_KSL_H
