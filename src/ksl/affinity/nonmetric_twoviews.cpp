#include <ksl/affinity/nonmetric_twoviews.h>
#include <iostream>

namespace ksl
{

namespace affinity
{

template<typename T>
NonMetricTwoViews<T>::NonMetricTwoViews(void)
{}

template<typename T>
NonMetricTwoViews<T>::NonMetricTwoViews(
  const int& nEig,
  const T& thresh, const T& eps,
  const bool& useNeg, const bool& remNoise):
  params_(nEig, thresh, eps, useNeg, remNoise)
{}

template<typename T>
NonMetricTwoViews<T>::NonMetricTwoViews(
  const NonMetricTwoViewsParams<T>& params):
  params_(params)
{}

template<typename T>
NonMetricTwoViews<T>::~NonMetricTwoViews(void)
{}

template<typename T>
void
NonMetricTwoViews<T>::computeAffinity(
  const mtxT& dMtx,
  const vecI& indVec)
{
  assert(dMtx.rows()==dMtx.cols());

  //t_.tic();

  if(params_.nEig>this->bparams_.nPoints)
  {
    params_.nEig=0;
  }

  mtxT hMtx(this->bparams_.nPoints, this->bparams_.nPoints);
  hMtx.setConstant(-1.0/this->bparams_.nPoints);
  hMtx.diagonal().array()+=1.0;
  mtxT cMtx;
  cMtx.noalias()=-0.5*hMtx*dMtx.array().square().matrix()*hMtx;

  computeTwoViews(cMtx);

  if(lVec_.size()>0)
  {
    delta_=std::pow(std::exp(-2.0)*dMtx.mean(), 2.0);
    computeAffinity();
    if(params_.remNoise)
    {
      if(params_.useNeg)
      {
        lVec_=(lVec_.array()>0.0).select(lVec_, 0.0);
      }
      removeNoise();
    }
  }

  //tAffinity_=t_.toc();
}

template<typename T>
void
NonMetricTwoViews<T>::computeAffinity(void)
{
  const int posSize=xposMtx_.rows(), negSize=xnegMtx_.rows();
  this->wMtx_.setZero(this->bparams_.nPoints, this->bparams_.nPoints);
  if(posSize>0)
  {
    wposMtx_.setZero(this->bparams_.nPoints, this->bparams_.nPoints);
    for(int i=0; i<this->bparams_.nPoints; ++i)
    {
      for(int j=i+1; j<this->bparams_.nPoints; ++j)
      {
        wposMtx_(i, j)=std::exp(-(xposMtx_.col(i)-xposMtx_.col(j)).squaredNorm()/delta_);
        wposMtx_(j, i)=wposMtx_(i, j);
      }
    }
    vecT dinvVec((wposMtx_.rowwise().sum().array()+T(1.0e-6)).pow(-0.5));
    this->wMtx_+=alpha_*dinvVec.asDiagonal()*wposMtx_*dinvVec.asDiagonal();
  }
  if(negSize>0)
  {
    wnegMtx_.setZero(this->bparams_.nPoints, this->bparams_.nPoints);
    for(int i=0; i<this->bparams_.nPoints; ++i)
    {
      for(int j=i+1; j<this->bparams_.nPoints; ++j)
      {
        wnegMtx_(i, j)=std::exp(-(xnegMtx_.col(i)-xnegMtx_.col(j)).squaredNorm()/delta_);
        wnegMtx_(j, i)=wnegMtx_(i, j);
      }
    }
    vecT dinvVec((wnegMtx_.rowwise().sum().array()+T(1.0e-6)).pow(-0.5));
    this->wMtx_+=(1.0-alpha_)*dinvVec.asDiagonal()*wnegMtx_*dinvVec.asDiagonal();
  }
}

template<typename T>
void
NonMetricTwoViews<T>::computeTwoViews(
  const mtxT& mtx)
{
  assert(mtx.rows()==mtx.cols());

  if(params_.nEig>0)
  {
    if(params_.useNeg)
    {
      Spectra::DenseSymMatProd<T> op(mtx);
      Spectra::SymEigsSolver<T, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<T> > eig(&op, params_.nEig, 2*params_.nEig+1);
      eig.init();
      try
      {
        eig.compute(1000, params_.eps);
        if(eig.info()!=Spectra::SUCCESSFUL)
        {
          std::cout<<"[Non-metric View using neg] eigendecomposition not successful"<<std::endl;
          vMtx_.resize(0, 0);
          lVec_.resize(0);
        }
        else
        {
          vMtx_=eig.eigenvectors();
          lVec_=eig.eigenvalues();
        }
      }
      catch(const std::exception& e)
      {
        std::cout<<"[Non-metric View using neg] "<<e.what()<<std::endl;
        vMtx_.resize(0, 0);
        lVec_.resize(0);
      }
    }
    else
    {
      Spectra::DenseSymMatProd<T> op(mtx);
      Spectra::SymEigsSolver<T, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<T> > eig(&op, params_.nEig, 2*params_.nEig+1);
      eig.init();
      try
      {
        eig.compute(1000, params_.eps);
        if(eig.info()!=Spectra::SUCCESSFUL)
        {
          std::cout<<"[Non-metric View] eigendecomposition not successful"<<std::endl;
          vMtx_.resize(0, 0);
          lVec_.resize(0);
        }
        else
        {
          vMtx_=eig.eigenvectors();
          lVec_=eig.eigenvalues();
        }
      }
      catch(const std::exception& e)
      {
        std::cout<<"[Non-metric View] "<<e.what()<<std::endl;
        vMtx_.resize(0, 0);
        lVec_.resize(0);
      }
    }
  }
  else
  {
    Eigen::SelfAdjointEigenSolver<mtxT> eig(mtx);
    vMtx_=eig.eigenvectors();
    lVec_=eig.eigenvalues();
  }

  std::vector<int> posInd, negInd;
  for(int i=0; i<lVec_.size(); ++i)
  {
    if(lVec_(i)>params_.eps)
    {
      posInd.push_back(i);
    }
    else if(lVec_(i)<-params_.eps)
    {
      negInd.push_back(i);
    }
  }

  const int posSize=posInd.size(), negSize=negInd.size();
  vecT lposVec(posSize), lnegVec(negSize);
  mtxT vposMtx(this->bparams_.nPoints, posSize), vnegMtx(this->bparams_.nPoints, negSize);
  std::vector<int>::iterator it=posInd.begin();
  for(int i=0; i<posSize; ++i, ++it)
  {
    vposMtx.col(i)=vMtx_.col(*it);
    lposVec(i)=lVec_(*it);
  }
  it=negInd.begin();
  for(int i=0; i<negSize; ++i, ++it)
  {
    vnegMtx.col(i)=vMtx_.col(*it);
    lnegVec(i)=-lVec_(*it);
  }

  T lposSum=0.0, lnegSum=0.0;
  T lambdaMax=0.0, lambdaMin=0.0;
  xposMtx_.resize(posSize, this->bparams_.nPoints), xnegMtx_.resize(negSize, this->bparams_.nPoints);
  if(posSize>0)
  {
    xposMtx_.noalias()=lposVec.cwiseSqrt().asDiagonal()*vposMtx.transpose();
    lposSum=lposVec.sum();
    lambdaMax=lposVec.maxCoeff();
  }
  if(negSize>0)
  {
    xnegMtx_.noalias()=lnegVec.cwiseSqrt().asDiagonal()*vnegMtx.transpose();
    lnegSum=lnegVec.sum();
    lambdaMin=lnegVec.maxCoeff();
  }

  alpha_=(lambdaMin<lambdaMax*params_.thresh) ? 1.0 : lposSum/(lposSum+lnegSum);
}

template<typename T>
void
NonMetricTwoViews<T>::removeNoise(void)
{
  vecT onesVec(this->bparams_.nPoints);
  onesVec.setOnes();
  mtxT kposMtx;
  kposMtx.noalias()=vMtx_*lVec_.asDiagonal();
  kposMtx*=vMtx_.transpose();
  rdMtx_.noalias()=kposMtx.diagonal()*onesVec.transpose();
  rdMtx_.noalias()+=onesVec*kposMtx.diagonal().transpose();
  rdMtx_-=2.0*kposMtx;
  rdMtx_=(rdMtx_.array()>0.0).select(rdMtx_.cwiseSqrt(), 0.0);
}

template class NonMetricTwoViews<double>;
template class NonMetricTwoViews<float>;

}

}
