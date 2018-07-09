#ifndef OFFLINE_SF_KSL_HPP
#define OFFLINE_SF_KSL_HPP

#include <assert.h>
#include <Eigen/Core>
#include <ksl/clustering/sparse_subspace_clustering.hpp>
#include <ksl/flow/pd_scene_flow.hpp>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace ksl
{

namespace offline_ksl
{

const float FX=525.5;
const float FY=525.5;

const float SUB_SAMPLING_VAL=10e-3;
const Eigen::Vector4f SUB_SAMPLING_LEAF_SIZE(
  SUB_SAMPLING_VAL, SUB_SAMPLING_VAL, SUB_SAMPLING_VAL, 0.0);

template<typename pointT>
class KinematicStructureLearning
{

private:

protected:

  int width_, height_;
  int rows_;
  int frames_, frame_;
  int nPointsSubSampled_;

  float cx_, cy_;
  int outlier_;
  float rho_;
  float fx_, fy_;
  float ratioHeightRows_;

  Eigen::VectorXi lostPointsInd_;

  typename pcl::PointCloud<pointT>::Ptr pcloud_, pcloudSubSampled_;
  Eigen::MatrixXf dataPoints_;
  flow::PDSceneFlow sceneFlow_;
  clustering::SparseSubspaceClustering subspaceClustering_;

public:

  KinematicStructureLearning(
    const int width, const int height,
    const int rows,
    const int frames,
    const float cx, const float cy,
    const int outlier=0,
    const float rho=1.0,
    const float fx=FX, const float fy=FY);
  virtual ~KinematicStructureLearning(void);

  const typename pcl::PointCloud<pointT>::ConstPtr
  pcloud(void) const
  {
    return pcloud_;
  }
  const typename pcl::PointCloud<pointT>::ConstPtr
  pcloudSubSampled(void) const
  {
    return pcloudSubSampled_;
  }
  int
  nGroups(void) const
  {
    return subspaceClustering_.nGroups();
  }
  const Eigen::VectorXi&
  groups(void) const
  {
    return subspaceClustering_.groups();
  }

  virtual float
  compute(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const cv::Mat &depth1,
    const cv::Mat &depth2,
    const float nPointsSubSampled=1500);

protected:

  virtual void
  depthToPCloud(
    const cv::Mat &img,
    const cv::Mat &depth)=0;

  virtual void
  mapDataPoints(void)=0;
  virtual void
  resizeDataPoints(void)=0;

  void
  fastBilateralFilter(
    const float sigmaS=5.0,
    const float sigmaR=5e-3);

  void
  subSamplePCloud(
    const Eigen::Vector4f &subSamplingLeafSize=SUB_SAMPLING_LEAF_SIZE);
  void
  subSamplePCloud(
    const int nSample);

  void
  ijToXYZ(
    const int i, const int j,
    const float depth,
    float *x, float *y, float *z);
  void
  xyzToIJ(
    const float x, const float y, const float z,
    int *i, int *j);

private:

};

class KinematicStructureLearningXYZ : public KinematicStructureLearning<pcl::PointXYZ>
{

private:

protected:

public:

  KinematicStructureLearningXYZ(
    const int width, const int height,
    const int rows,
    const int frames);
  KinematicStructureLearningXYZ(
    const int width, const int height,
    const int rows,
    const int frames,
    const float cx, const float cy,
    const int outlier=0,
    const float rho=1.0,
    const float fx=FX, const float fy=FY);
  ~KinematicStructureLearningXYZ(void);

protected:

  void
  depthToPCloud(
    const cv::Mat &img,
    const cv::Mat &depth);

  void
  mapDataPoints(void);
  void
  resizeDataPoints(void);

private:

};

class KinematicStructureLearningXYZRGB : public KinematicStructureLearning<pcl::PointXYZRGB>
{

private:

protected:

public:

  KinematicStructureLearningXYZRGB(
    const int width, const int height,
    const int rows,
    const int frames);
  KinematicStructureLearningXYZRGB(
    const int width, const int height,
    const int rows,
    const int frames,
    const float cx, const float cy,
    const int outlier=0,
    const float rho=1.0,
    const float fx=FX, const float fy=FY);
  ~KinematicStructureLearningXYZRGB(void);

protected:

  void
  depthToPCloud(
    const cv::Mat &img,
    const cv::Mat &depth);

  void
  mapDataPoints(void);
  void
  resizeDataPoints(void);

private:

};

}

}

#endif // OFFLINE_SF_KSL_HPP

