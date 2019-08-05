#ifndef PCLOUD_H
#define PCLOUD_H

#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>
#include <opencv2/opencv.hpp>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace ksl
{

namespace utils
{

template<typename T>
struct CamParams
{

  int width, height;
  T cx, cy;
  T fx, fy;

  CamParams(void):
    width(-1), height(-1),
    cx(319.5), cy(239.5),
    fx(525.5), fy(525.5)
  {}
  CamParams(
    const int& width, const int& height):
    width(width), height(height),
    cx(319.5), cy(239.5),
    fx(525.5), fy(525.5)
  {}
  CamParams(
    const int& width, const int& height,
    const T& cx, const T& cy,
    const T& fx, const T& fy):
    width(width), height(height),
    cx(cx), cy(cy),
    fx(fx), fy(fy)
  {}
  CamParams(
    const CamParams& params):
    width(params.width), height(params.height),
    cx(params.cx), cy(params.cy),
    fx(params.fx), fy(params.fy)
  {}
  ~CamParams(void)
  {}

  CamParams<T>&
  operator=(
    const CamParams<T>& params)
  {
    width=params.width;
    height=params.height;
    cx=params.cx;
    cy=params.cy;
    fx=params.fx;
    fy=params.fy;
    return *this;
  }

};

template<typename T>
inline void
ijToXYZ(
  const int& i, const int& j,
  const T& depth,
  const CamParams<T>& params,
  T* x, T* y, T* z)
{
  if(depth>0.0)
  {
    *z=depth;
    *x=*z*(j-params.cx)/params.fx;
    *y=*z*(params.cy-i)/params.fy;
  }
  else
  {
    *x=std::numeric_limits<T>::quiet_NaN();
    *y=std::numeric_limits<T>::quiet_NaN();
    *z=std::numeric_limits<T>::quiet_NaN();
  }
}

template<typename T>
inline void
xyzToIJ(
  const T& x, const T& y, const T& z,
  const CamParams<T>& params,
  int* i, int* j)
{
  if(z>0.0)
  {
    *j=std::round(params.cx+(params.fx*x/z));
    *i=std::round(params.cy-(params.fy*y/z));
  }
  else
  {
    *i=-1;
    *j=-1;
  }
}

template<typename T>
inline void
depthToPCloud(
  const cv::Mat& img,
  const cv::Mat& depth,
  const CamParams<T>& params,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& pcloud)
{
  assert(img.cols==depth.cols && img.rows==depth.rows);
  assert(img.cols==pcloud->width && img.rows==pcloud->height);

  const int width=img.cols, height=img.rows;

  pcl::PointXYZ pt;
  pcloud->is_dense=false;
  for(int i=0; i<height; ++i)
  {
    for(int j=0; j<width; ++j)
    {
      ijToXYZ(i, j, depth.at<T>(i, j), params, &pt.x, &pt.y, &pt.z);
      pcloud->at(j, height-1-i)=pt;
    }
  }
}

template<typename T>
inline void
depthToPCloud(
  const cv::Mat& img,
  const cv::Mat& depth,
  const CamParams<T>& params,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcloud)
{
  assert(img.cols==depth.cols && img.rows==depth.rows);
  assert(img.cols==pcloud->width && img.rows==pcloud->height);

  const int width=img.cols, height=img.rows;

  pcl::PointXYZRGB pt;
  cv::Vec3b pixel;
  pcloud->is_dense=false;
  for(int i=0; i<height; ++i)
  {
    for(int j=0; j<width; ++j)
    {
      ijToXYZ(i, j, depth.at<T>(i, j), params, &pt.x, &pt.y, &pt.z);
      pixel=img.at<cv::Vec3b>(i, j);
      pt.b=pixel.val[0];
      pt.g=pixel.val[1];
      pt.r=pixel.val[2];
      pcloud->at(j, height-1-i)=pt;
    }
  }
}

template<typename T, typename pointT>
inline void
fastBilateralFilter(
  const T& sigmaS, const T& sigmaR,
  typename pcl::PointCloud<pointT>::Ptr& pcloud)
{
  typename pcl::FastBilateralFilterOMP<pointT> fbf;
  fbf.setSigmaS(sigmaS);
  fbf.setSigmaR(sigmaR);
  fbf.setInputCloud(pcloud);
  fbf.applyFilter(*pcloud);
}

template<typename T, typename pointT>
inline void
fastBilateralFilter(
  const T& sigmaS, const T& sigmaR,
  const typename pcl::PointCloud<pointT>::ConstPtr& pcloudOriginal,
  typename pcl::PointCloud<pointT>::Ptr& pcloud)
{
  typename pcl::FastBilateralFilterOMP<pointT> fbf;
  fbf.setSigmaS(sigmaS);
  fbf.setSigmaR(sigmaR);
  fbf.setInputCloud(pcloudOriginal);
  fbf.applyFilter(*pcloud);
}

template<typename T, typename pointT>
inline void
subSamplePCloud(
  const typename pcl::PointCloud<pointT>::ConstPtr& pcloudOriginal,
  const int& nPoints,
  typename pcl::PointCloud<pointT>::Ptr& pcloudSubSampled,
  const unsigned int& seedNum=std::time(nullptr))
{
  std::vector<int> indVec;
  pcl::removeNaNFromPointCloud(*pcloudOriginal, *pcloudSubSampled, indVec);
  typename pcl::RandomSample<pointT> subSamplingFilter;
  subSamplingFilter.setSeed(seedNum);
  subSamplingFilter.setSample(nPoints);
  subSamplingFilter.setInputCloud(pcloudSubSampled);
  subSamplingFilter.filter(*pcloudSubSampled);
}

template<typename T, typename pointT>
inline void
subSamplePCloud(
  const typename pcl::PointCloud<pointT>::ConstPtr& pcloudOriginal,
  const int& nPoints,
  typename pcl::PointCloud<pointT>::Ptr& pcloud,
  typename pcl::PointCloud<pointT>::Ptr& pcloudSubSampled,
  const unsigned int& seedNum=std::time(nullptr))
{
  std::vector<int> indVec;
  pcl::removeNaNFromPointCloud(*pcloudOriginal, *pcloud, indVec);
  typename pcl::RandomSample<pointT> subSamplingFilter;
  subSamplingFilter.setSeed(seedNum);
  subSamplingFilter.setSample(nPoints);
  subSamplingFilter.setInputCloud(pcloud);
  subSamplingFilter.filter(*pcloudSubSampled);
}

template<typename T, typename pointT>
inline void
subSamplePCloud(
  const typename pcl::PointCloud<pointT>::ConstPtr& pcloudOriginal,
  const T& resolution,
  typename pcl::PointCloud<pointT>::Ptr& pcloudSubSampled)
{
  std::vector<int> indVec;
  pcl::removeNaNFromPointCloud(*pcloudOriginal, *pcloudSubSampled, indVec);
  typename pcl::VoxelGrid<pointT> subSamplingFilter;
  subSamplingFilter.setLeafSize(resolution, resolution, resolution);
  subSamplingFilter.setInputCloud(pcloudSubSampled);
  subSamplingFilter.filter(*pcloudSubSampled);
}

template<typename T, typename pointT>
inline void
subSamplePCloud(
  const typename pcl::PointCloud<pointT>::ConstPtr& pcloudOriginal,
  const T& resolution,
  typename pcl::PointCloud<pointT>::Ptr& pcloud,
  typename pcl::PointCloud<pointT>::Ptr& pcloudSubSampled)
{
  std::vector<int> indVec;
  pcl::removeNaNFromPointCloud(*pcloudOriginal, *pcloud, indVec);
  typename pcl::VoxelGrid<pointT> subSamplingFilter;
  subSamplingFilter.setLeafSize(resolution, resolution, resolution);
  subSamplingFilter.setInputCloud(pcloud);
  subSamplingFilter.filter(*pcloudSubSampled);
}

}

}

#endif // PCLOUD_H
