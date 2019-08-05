#include <ksl/clustering/spectral_clustering.h>

namespace ksl
{
  
namespace clustering
{

template<typename T>
SpectralClustering<T>::SpectralClustering(void)
{}

template<typename T>
SpectralClustering<T>::SpectralClustering(
  const T& eigThresh, const T& eps, const T& thresh,
  const int& nMaxClusters, const int& nIter):
  params_(eigThresh, eps, thresh, nMaxClusters, nIter)
{
  assert(params_.nMaxClusters>0);
}

template<typename T>
SpectralClustering<T>::SpectralClustering(
  const SpectralClusteringParams<T>& params):
  params_(params)
{
  assert(params_.nMaxClusters>0);
}

template<typename T>
SpectralClustering<T>::~SpectralClustering(void)
{}

template<typename T>
void
SpectralClustering<T>::computeClusters(
  const mtxT& dMtx)
{
  assert(this->affinity_!=nullptr);
  
  if(this->bparams_.nClusters<=0)
  {
    nEig_=(this->bparams_.nPoints>=40) ? params_.nMaxClusters : 0.5*this->bparams_.nPoints;
    this->bparams_.nClusters=0;
  }
  else
  {
    if(this->bparams_.nClusters==1)
    {
      this->clusters_.setZero();
      return ;
    }
    nEig_=this->bparams_.nClusters;
  }

  this->affinity_->compute(dMtx);
  Eigen::Map<const mtxT> wMtx(this->affinity_->wMtx().data(), this->bparams_.nPoints, this->bparams_.nPoints);

  mtxT lapMtx;
  laplacian<T>(wMtx, lapMtx);

  Spectra::DenseSymMatProd<T> op(lapMtx);
  Spectra::SymEigsSolver<T, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<T> > eig(&op, nEig_, 2*nEig_);
  eig.init();
  eig.compute(1000, params_.eps);
  const mtxT vMtx(eig.eigenvectors());

  if(this->bparams_.nClusters==0)
  {
    const vecT lVec(eig.eigenvalues());
    while(lVec(nEig_-1)<params_.eigThresh && nEig_>1)
    {
      --nEig_;
    }
    if(nEig_==1)
    {
      this->bparams_.nClusters=1;
      this->clusters_.setZero();
      return ;
    }

    rotateClusters(vMtx);
  }
  else
  {
    KMeans<T> km(1, 100);
    km.compute(vMtx, this->bparams_.nClusters);
    this->clusters_=km.clusters();
  }
}

template<typename T>
void
SpectralClustering<T>::buildU(
  const vecT& theta,
  const vecT& ik, const vecT& jk,
  const int& a, const int& b,
  mtxT& uMtx) const
{
  uMtx.setIdentity();

  T cosTheta, sinTheta, uik;
  for(int k=a; k<b; ++k)
  {
    cosTheta=std::cos(theta(k)), sinTheta=std::sin(theta(k));
    for(int i=0; i<nDims_; ++i)
    {
      uik=uMtx(i, ik(k))*cosTheta-uMtx(i, jk(k))*sinTheta;
      uMtx(i, jk(k))=uMtx(i, ik(k))*sinTheta+uMtx(i, jk(k))*cosTheta;
      uMtx(i, ik(k))=uik;
    }
  }
}

template<typename T>
T
SpectralClustering<T>::evaluateQuality(
  const mtxT& vMtx) const
{
  mtxT vSquaredMtx(vMtx.array().square());
  vecT vMaxVec(vSquaredMtx.rowwise().maxCoeff());

  for(int i=0; i<this->bparams_.nPoints; ++i)
  {
    if(vMaxVec(i)!=0.0)
    {
      vSquaredMtx.row(i)/=vMaxVec(i);
    }
  }
  return 1.0-(vSquaredMtx.sum()/this->bparams_.nPoints-1.0)/nDims_;
}

template<typename T>
T
SpectralClustering<T>::evaluateQualityGradient(
  const mtxT& vMtx,
  const vecT& theta,
  const vecT& ik, const vecT& jk,
  const int& ind)
{
  gradientTheta(theta, ik, jk, ind);
  buildU(theta, ik, jk, 0, ind, u1Mtx_);
  buildU(theta, ik, jk, ind+1, nAngles_, u2Mtx_);

  mtxT aMtx, vRotMtx;
  aMtx.noalias()=vMtx*u1Mtx_;
  aMtx*=uMtx_, aMtx*=u2Mtx_;

  rotate(vMtx, theta, ik, jk, vRotMtx);

  vecT vMaxVec(this->bparams_.nPoints);
  vecI indMaxVec(this->bparams_.nPoints);
  for(int i=0; i<this->bparams_.nPoints; ++i)
  {
    vRotMtx.row(i).cwiseAbs().maxCoeff(&indMaxVec(i));
    vMaxVec(i)=vRotMtx(i, indMaxVec(i));
  }

  /* gradient computation */
  T dA=0.0, tmp1, tmp2, tmp3;
  for(int j=0; j<nDims_; ++j)
  {
    for(int i=0; i<this->bparams_.nPoints; ++i)
    {
      if(vMaxVec(i)!=0.0)
      {
        tmp3=vMaxVec(i)*vMaxVec(i);
        tmp1=aMtx(i, j)*vRotMtx(i, j)/tmp3;
        tmp2=aMtx(i, indMaxVec(i))*(vRotMtx(i, j)*vRotMtx(i, j))/(tmp3*vMaxVec(i));
        dA+=tmp1-tmp2;
      }
    }
  }
  return dA*2.0/(this->bparams_.nPoints*nDims_);
}

template<typename T>
void
SpectralClustering<T>::gradientTheta(
  const vecT& theta,
  const vecT& ik, const vecT& jk,
  const int& ind)
{
  const T cosTheta=std::cos(theta(ind)), sinTheta=std::sin(theta(ind));
  uMtx_(ik(ind), ik(ind))=-sinTheta;
  uMtx_(ik(ind), jk(ind))=cosTheta;
  uMtx_(jk(ind), ik(ind))=-cosTheta;
  uMtx_(jk(ind), jk(ind))=-sinTheta;
}

template<typename T>
void
SpectralClustering<T>::rotate(
  const mtxT& vMtx,
  const vecT& theta,
  const vecT& ik, const vecT& jk,
  mtxT& vRotMtx)
{
  buildU(theta, ik, jk, 0, nAngles_, uMtx_);
  vRotMtx.noalias()=vMtx*uMtx_;
}

template<typename T>
T
SpectralClustering<T>::rotateClusters(
  mtxT& vMtx,
  const T& rotQualMax)
{
  nDims_=vMtx.cols();
  nAngles_=nDims_*(nDims_-1)*0.5;
  uMtx_.setZero(nDims_, nDims_), u1Mtx_.resize(nDims_, nDims_), u2Mtx_.resize(nDims_, nDims_);

  vecT curTheta(nAngles_), newTheta(nAngles_);
  curTheta.setZero(), newTheta.setZero();
  vecT ik(nAngles_), jk(nAngles_);
  int k=0;
  for(int i=0; i<nDims_-1; ++i)
  {
    for(int j=i+1; j<nDims_; ++j)
    {
      ik(k)=i, jk(k)=j, ++k;
    }
  }

  mtxT vBestRotMtx(vMtx), vRotMtx;
  /* initial quality */
  T curRotQual=evaluateQuality(vMtx), newRotQual;
  T dRotQual;
  T rotQual1=curRotQual, rotQual2=curRotQual;

  /* quality refinement */
  for(int k=0; k<params_.nIter; ++k)
  {
    for(int j=0; j<nAngles_; ++j)
    {
      dRotQual=evaluateQualityGradient(vMtx, curTheta, ik, jk, j);
      newTheta(j)=curTheta(j)-dRotQual;
      rotate(vMtx, newTheta, ik, jk, vRotMtx);
      newRotQual=evaluateQuality(vRotMtx);

      if(newRotQual>curRotQual)
      {
        curTheta(j)=newTheta(j);
        curRotQual=newRotQual;
        vBestRotMtx=vRotMtx;
      }
      else
      {
        newTheta(j)=curTheta(j);
      }
    }

    /* stopping criteria */
    if(k>2 && curRotQual-rotQual2<1.0e-3)
    {
      break;
    }
    rotQual2=rotQual1;
    rotQual1=curRotQual;
  }

  vMtx=vBestRotMtx;
  if(qualCondition(curRotQual, rotQualMax))
  {
    for(int i=0; i<this->bparams_.nPoints; ++i)
    {
      vMtx.row(i).cwiseAbs().maxCoeff(&this->clusters_(i));
    }
  }

  return curRotQual;
}

template<typename T>
void
SpectralClustering<T>::rotateClusters(
  const mtxT& vMtx)
{
  mtxT vRotMtx(vMtx.leftCols(2)), vRotMaxMtx;
  T rotQual, rotQualMax=rotateClusters(vRotMtx, -1.0);
  vRotMaxMtx=vRotMtx;

  //std::cout<<2<<"/"<<nEig_<<" "<<rotQualMax<<std::endl;

  this->bparams_.nClusters=2;
  for(int i=3; i<=nEig_; ++i)
  {
    vRotMtx.conservativeResize(Eigen::NoChange, i);
    vRotMtx.col(i-1)=vMtx.col(i-1);

    // clusters are automatically updated if quality condition is met
    rotQual=rotateClusters(vRotMtx, rotQualMax);
    //std::cout<<i<<"/"<<nEig_<<" "<<rotQual<<std::endl;
    if(qualCondition(rotQual, rotQualMax))
    {
      rotQualMax=rotQual;
      vRotMaxMtx=vRotMtx;
      this->bparams_.nClusters=i;
    }
  }

  KMeans<T> km(2, 100);
  km.compute(vRotMaxMtx, this->clusters_, this->bparams_.nClusters);
  this->clusters_=km.clusters();
}

template class SpectralClustering<double>;
template class SpectralClustering<float>;

}

}
