#ifndef PD_SCENE_FLOW_H
#define PD_SCENE_FLOW_H

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ksl/flow/pd_scene_flow_cuda.h>
#include <opencv2/opencv.hpp>

namespace ksl
{

namespace flow
{

class PDSceneFlow
{

private:

  const int width_, height_;
  int camMode_;

  const int rows_, cols_, size_;
  const int coarseToFineLvls_;
  int maxNIter_[6];
  float gaussianMask_[25];

  float* imgPointer_;
  float* depthPointer_;

  CSF_cuda csfHost_;
  CSF_cuda* csfDevice_;

protected:

  float* dxp_;
  float* dyp_;
  float* dzp_;

  cv::Mat imgSceneFlow_;

public:

  PDSceneFlow(
    const int& width, const int& height,
    const int& rows=240,
    const float& fovh=M_PI*62.5/180.0, const float& fovv=M_PI*48.5/180.0);
  ~PDSceneFlow(void);

  inline const int&
  width(void) const
  {
    return width_;
  }
  inline const int&
  height(void) const
  {
    return height_;
  }
	inline const int&
  rows(void) const
  {
    return rows_;
  }
	inline const int&
  cols(void) const
  {
    return cols_;
  }
  inline const int&
  size(void) const
  {
    return size_;
  }
	inline float*
  dxp(void)
  {
    return dxp_;
  }
  inline float*
  dyp(void)
  {
    return dyp_;
  }
  inline float*
  dzp(void)
  {
    return dzp_;
  }
  inline const float*
  dxp(void) const
  {
    return dxp_;
  }
  inline const float*
  dyp(void) const
  {
    return dyp_;
  }
  inline const float*
  dzp(void) const
  {
    return dzp_;
  }
  inline const cv::Mat&
  imgSceneFlow(void) const
  {
    return imgSceneFlow_;
  }

  void
  compute(
    const cv::Mat& img1, const cv::Mat& img2,
    const cv::Mat& depth1, const cv::Mat& depth2);

  void
  createImg(void);

  void
  showImg(void) const;

protected:

  void
  init(void);
  void
  initCuda(
    const float& fovh, const float& fovv);

  void
  createImgPyramidGPU(void);
  void
  createImgPyramidGPU(
    const cv::Mat& img1, const cv::Mat& img2,
    const cv::Mat& depth1, const cv::Mat& depth2);

private:

};

}

}

#endif // PD_SCENE_FLOW_H
