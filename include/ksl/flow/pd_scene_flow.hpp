#ifndef PD_SCENE_FLOW_HPP
#define PD_SCENE_FLOW_HPP

#include <assert.h>
#include <Eigen/Core>
#include <math.h>
#include <ksl/flow/pd_scene_flow_cuda.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

namespace ksl
{

namespace flow
{

class PDSceneFlow
{

private:

  int width_;
  int height_;
  int camMode_;

  int rows_, cols_, size_;
  int coarseToFineLvls_;
  int maxNIter_[6];
  float gaussianMask_[25];

  float *imgPointer_;
  float *depthPointer_;

  CSF_cuda csfHost_;
  CSF_cuda *csfDevice_;

protected:

  float *dxp_;
  float *dyp_;
  float *dzp_;

  cv::Mat imgSceneFlow_;

public:

  PDSceneFlow(
    const int width,
    const int height,
    const int rows=240);
  virtual ~PDSceneFlow(void);

  const int
  size(void) const
  {
    return size_;
  }
  const float*
  dxp(void) const
  {
    return dxp_;
  }
  const float*
  dyp(void) const
  {
    return dyp_;
  }
  const float*
  dzp(void) const
  {
    return dzp_;
  }

  const cv::Mat&
  imgSceneFlow(void) const
  {
    return imgSceneFlow_;
  }

  virtual void
  compute(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const cv::Mat &depth1,
    const cv::Mat &depth2);

  virtual void
  createImg(void);

  virtual void
  showImg(void) const;

protected:

  virtual void
  init(void);
  virtual void
  initCuda(void);

  virtual void
  createImgPyramidGPU(void);
  virtual void
  createImgPyramidGPU(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const cv::Mat &depth1,
    const cv::Mat &depth2);

private:

};

}

}

#endif // PD_SCENE_FLOW_HPP

