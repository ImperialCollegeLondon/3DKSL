#include <ksl/flow/pd_scene_flow.h>

namespace ksl
{

namespace flow
{

PDSceneFlow::PDSceneFlow(
  const int& width, const int& height,
  const int& rows,
  const float& fovh, const float& fovv):
  width_(width), height_(height),
  rows_(rows), cols_(rows_*320.0/240.0), size_(rows_*cols_),
  coarseToFineLvls_(std::log2((float) rows_/15.0)+1),
  imgSceneFlow_(rows_, cols_, CV_8UC3)
{
  init();
  initCuda(fovh, fovv);
}

PDSceneFlow::~PDSceneFlow(void)
{
  csfHost_.freeDeviceMemory();
  free(dxp_);
  free(dyp_);
  free(dzp_);
  free(imgPointer_);
  free(depthPointer_);
}

void
PDSceneFlow::compute(
  const cv::Mat& img1, const cv::Mat& img2,
  const cv::Mat& depth1, const cv::Mat& depth2)
{
  assert(img1.cols==width_ && img2.cols==width_ && depth1.cols ==width_ && depth2.cols==width_);
  assert(img1.rows==height_ && img2.rows==height_ && depth1.rows ==height_ && depth2.rows==height_);
  assert(img1.channels()==1 && img2.channels()==1 && depth1.channels()==1 && depth2.channels()==1);

  createImgPyramidGPU(img1, img2, depth1, depth2);

  int s, cols, rows, imgLvl;
  /* solve scene flow using GPU */
  for(int i=0; i<coarseToFineLvls_; ++i)
  {
    s=std::pow(2.0, coarseToFineLvls_-(i+1));
    cols=cols_/s;
    rows=rows_/s;
    imgLvl=coarseToFineLvls_-i+std::log2((float) width_/cols_)-1;

    /* Cuda memory allocation */
    csfHost_.allocateMemoryNewLevel(rows, cols, i, imgLvl);
    /* Cuda copy of object to device */
    csfDevice_=ObjectToDevice(&csfHost_);
    AssignZerosBridge(csfDevice_);
    if(i>0)
    {
      /* upsample previous solution */
      UpsampleBridge(csfDevice_);
    }
    /* connectivity computation */
    RijBridge(csfDevice_);
    /* color and depth derivatives computation */
    ImageGradientsBridge(csfDevice_);
    WarpingBridge(csfDevice_);
    /* mu_uv and step sizes computation for the primal-dual algorithm */
    MuAndStepSizesBridge(csfDevice_);
    /* primal-dual solver */
    for(int j=0; j<maxNIter_[i]; ++j)
    {
      GradientBridge(csfDevice_);
      DualVariablesBridge(csfDevice_);
      DivergenceBridge(csfDevice_);
      PrimalVariablesBridge(csfDevice_);
    }
    /* filter solution */
    FilterBridge(csfDevice_);
    /* motion field computation */
    MotionFieldBridge(csfDevice_);
    BridgeBack(&csfHost_, csfDevice_);
    /* free memory associated to this level */
    csfHost_.freeLevelVariables();
    csfHost_.copyMotionField(dxp_, dyp_, dzp_);
  }
}

void
PDSceneFlow::createImg(void)
{
  float maxdX=0.0, maxdY=0.0, maxdZ=0.0;
  /* maximum values of the scene flow per component */
  for(int i=0; i<rows_; ++i)
  {
    for(int j=0; j<cols_; ++j)
    {
      int jj=j*rows_;
      float valAbs=std::abs(dxp_[i+jj]);
      if(valAbs>maxdX)
      {
        maxdX=valAbs;
      }
      valAbs=std::abs(dyp_[i+jj]);
      if(valAbs>maxdY)
      {
        maxdY=valAbs;
      }
      valAbs=std::abs(dzp_[i+jj]);
      if(valAbs>maxdZ)
      {
        maxdZ=valAbs;
      }
    }
  }

  /* scene flow estimate representation */
  for(int i=0; i<rows_; ++i)
  {
    for(int j=0; j<cols_; ++j)
    {
      int jj=j*rows_;
      imgSceneFlow_.at<cv::Vec3b>(i, j)[0]=(unsigned char) 255.0*std::abs(dxp_[i+jj]/maxdX);
      imgSceneFlow_.at<cv::Vec3b>(i, j)[1]=(unsigned char) 255.0*std::abs(dyp_[i+jj]/maxdY);
      imgSceneFlow_.at<cv::Vec3b>(i, j)[2]=(unsigned char) 255.0*std::abs(dzp_[i+jj]/maxdZ);
    }
  }
}

void
PDSceneFlow::showImg(void) const
{
  cv::namedWindow("Scene Flow Estimation", cv::WINDOW_NORMAL);
  cv::moveWindow("Scene Flow Estimation", width_-cols_*0.5, height_-rows_*0.5);
  cv::imshow("Scene Flow Estimation", imgSceneFlow_);
  cv::waitKey(0);
}

void
PDSceneFlow::init(void)
{
  /* iterations of the primal-dual solver at each pyramid level */
  for(int i=5; i>=0; --i)
  {
    (i>=coarseToFineLvls_-1) ? (maxNIter_[i]=100) : (maxNIter_[i]=maxNIter_[i+1]-15);
  }

  /* gaussian mask */
  const int vMask[5]={1, 4, 6, 4, 1};
  for(int i=0; i<5; ++i)
  {
    for(int j=0; j<5; ++j)
    {
      gaussianMask_[i+j*5]=(float) (vMask[i]*vMask[j])/256.0;
    }
  }

  /* memory allocation for the scene flow estimation */
  dxp_=(float *) malloc(sizeof(float)*size_);
  dyp_=(float *) malloc(sizeof(float)*size_);
  dzp_=(float *) malloc(sizeof(float)*size_);
}

void
PDSceneFlow::initCuda(
  const float& fovh, const float& fovv)
{
  (height_==240) ? (camMode_=2) : (camMode_=1);
  imgPointer_=(float *) malloc(sizeof(float)*width_*height_);
  depthPointer_=(float *) malloc(sizeof(float)*width_*height_);

  csfHost_.readParameters(rows_, cols_, 0.04, 0.35, 75.0, gaussianMask_, coarseToFineLvls_,
    camMode_, fovh, fovv);
  csfHost_.allocateDevMemory();
}

void
PDSceneFlow::createImgPyramidGPU(void)
{
  csfHost_.copyNewFrames(imgPointer_, depthPointer_);
  csfDevice_=ObjectToDevice(&csfHost_);
  int pyramidLvls=std::log2((float) width_/cols_)+coarseToFineLvls_;
  GaussianPyramidBridge(csfDevice_, pyramidLvls, camMode_);
  BridgeBack(&csfHost_, csfDevice_);
}

void
PDSceneFlow::createImgPyramidGPU(
  const cv::Mat &img1, const cv::Mat &img2,
  const cv::Mat &depth1, const cv::Mat &depth2)
{
  /* first images */
  for(int i=0; i<width_; ++i)
  {
    int ii=i*height_;
    for(int j=0; j<height_; ++j)
    {
      imgPointer_[j+ii]=(float) img1.at<unsigned char>(j, i);
      depthPointer_[j+ii]=depth1.at<float>(j, i);
    }
  }
  createImgPyramidGPU();

  /* second images */
  for(int i=0; i<width_; ++i)
  {
    int ii=i*height_;
    for(int j=0; j<height_; ++j)
    {
      imgPointer_[j+ii]=(float) img2.at<unsigned char>(j, i);
      depthPointer_[j+ii]=depth2.at<float>(j, i);
    }
  }
  createImgPyramidGPU();
}

}

}
