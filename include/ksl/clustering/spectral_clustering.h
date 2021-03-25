#ifndef SPECTRAL_CLUSTERING_H
#define SPECTRAL_CLUSTERING_H

#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <ksl/clustering/clustering.h>
#include <ksl/clustering/kmeans.h>
#include <Spectra/SymEigsSolver.h>

namespace ksl
{

namespace clustering
{

template<typename T>
struct SpectralClusteringParams
{

  T eigThresh, eps, thresh;
  int nMaxClusters, nIter;

  SpectralClusteringParams(void):
    eigThresh(0.0), eps(1.0e-6), thresh(1.0e-3),
    nMaxClusters(20), nIter(200)
  {}
  SpectralClusteringParams(
    const T& eigThresh, const T& eps, const T& thresh,
    const int& nMaxClusters, const int& nIter):
    eigThresh(eigThresh), eps(eps), thresh(thresh),
    nMaxClusters(nMaxClusters), nIter(nIter)
  {}
  SpectralClusteringParams(
    const SpectralClusteringParams<T>& params):
    eigThresh(params.eigThresh), eps(params.eps), thresh(params.thresh),
    nMaxClusters(params.nMaxClusters), nIter(params.nIter)
  {}

  SpectralClusteringParams<T>&
  operator=(
    const SpectralClusteringParams<T>& params)
  {
    eigThresh=params.eigThresh;
    eps=params.eps;
    thresh=params.thresh;
    nMaxClusters=params.nMaxClusters;
    nIter=params.nIter;
    return *this;
  }

};

template<typename T>
class SpectralClustering : public Clustering<T>
{

public:

  typedef typename Clustering<T>::vecI vecI;
  typedef typename Clustering<T>::vecT vecT;
  typedef typename Clustering<T>::mtxT mtxT;

private:

  int nAngles_, nDims_;
  mtxT uMtx_, u1Mtx_, u2Mtx_;

protected:

  int nEig_;

  SpectralClusteringParams<T> params_;

public:

  SpectralClustering(void);
  SpectralClustering(
    const T& eigThresh, const T& eps, const T& thresh,
    const int& nMaxClusters, const int& nIter);
  SpectralClustering(
    const SpectralClusteringParams<T>& params);
  ~SpectralClustering(void);

  inline const int&
  nEig(void) const
  {
    return nEig_;
  }
  inline const T&
  eigThresh(void) const
  {
    return params_.eigThresh;
  }
  inline const T&
  eps(void) const
  {
    return params_.eps;
  }
  inline const T&
  thresh(void) const
  {
    return params_.thresh;
  }
  inline const int&
  nMaxClusters(void) const
  {
    return params_.nMaxClusters;
  }
  inline const int&
  nIter(void) const
  {
    return params_.nIter;
  }

protected:

  void
  computeClusters(
    const mtxT& dMtx);

  inline void
  buildU(
    const vecT& theta,
    const vecI& ik, const vecI& jk,
    const int& a, const int& b,
    mtxT& uMtx) const;

  inline T
  evaluateQuality(
    const mtxT& vMtx) const;
  inline T
  evaluateQualityGradient(
    const mtxT& vMtx,
    const vecT& theta,
    const vecI& ik, const vecI& jk,
    const int& ind);

  inline void
  gradientTheta(
    const vecT& theta,
    const vecI& ik, const vecI& jk,
    const int& ind);

  inline void
  rotate(
    const mtxT& vMtx,
    const vecT& theta,
    const vecI& ik, const vecI& jk,
    mtxT& vRotMtx);
  inline T
  rotateClusters(
    mtxT& vMtx,
    const T& rotQualMax);
  inline void
  rotateClusters(
    const mtxT& vMtx);

private:

  inline bool
  qualCondition(
    const T& rotQual,
    const T& rotQualMax) const
  {
    return (rotQual-rotQualMax>params_.thresh) || rotQual>=1.0;
  }

};

}

}

#endif // SPECTRAL_CLUSTERING_H
