#ifndef NONMETRIC_TWOVIEWS_H
#define NONMETRIC_TWOVIEWS_H

#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iterator>
#include <ksl/affinity/affinity.h>
//#include <ksl/utils/tictoc.h>
#include <Spectra/SymEigsSolver.h>
#include <vector>

namespace ksl
{

namespace affinity
{

template<typename T>
struct NonMetricTwoViewsParams
{

  int nEig;
  T thresh, eps;
  bool useNeg, remNoise;

  NonMetricTwoViewsParams(void):
    nEig(0),
    thresh(0.01), eps(1.0e-6),
    useNeg(true), remNoise(false)
  {}
  NonMetricTwoViewsParams(
    const int& nEig,
    const T& thresh, const T& eps,
    const bool& useNeg, const bool& remNoise):
    nEig(nEig),
    thresh(thresh), eps(eps),
    useNeg(useNeg), remNoise(remNoise)
  {}
  NonMetricTwoViewsParams(
    const NonMetricTwoViewsParams<T>& params):
    nEig(params.nEig),
    thresh(params.thresh), eps(params.eps),
    useNeg(params.useNeg), remNoise(params.remNoise)
  {}

};

template<typename T>
class NonMetricTwoViews : public Affinity<T>
{

public:

  typedef typename Affinity<T>::vecI vecI;
  typedef typename Affinity<T>::vecT vecT;
  typedef typename Affinity<T>::mtxI mtxI;
  typedef typename Affinity<T>::mtxT mtxT;

private:

  //utils::TicToc t_;
  //double tAffinity_;

  vecT lVec_;
  mtxT vMtx_;

protected:

  T alpha_, delta_;

  NonMetricTwoViewsParams<T> params_;

  mtxT xposMtx_, xnegMtx_;
  mtxT wposMtx_, wnegMtx_;
  mtxT rdMtx_;

public:

  NonMetricTwoViews(void);
  NonMetricTwoViews(
    const int& nEig,
    const T& thresh, const T& eps,
    const bool& useNeg, const bool& remNoise);
  NonMetricTwoViews(
    const NonMetricTwoViewsParams<T>& params);
  ~NonMetricTwoViews(void);

  /*inline const double&
  tAffinity(void) const
  {
    return tAffinity_;
  }*/

  inline const T&
  alpha(void) const
  {
    return alpha_;
  }
  inline const T&
  delta(void) const
  {
    return delta_;
  }
  inline const int&
  nEig(void) const
  {
    return params_.nEig;
  }
  inline const T&
  thresh(void) const
  {
    return params_.thresh;
  }
  inline const T&
  eps(void) const
  {
    return params_.eps;
  }
  inline const bool&
  useNeg(void) const
  {
    return params_.useNeg;
  }
  inline const bool&
  remNoise(void) const
  {
    return params_.remNoise;
  }
  inline const mtxT&
  xposMtx(void) const
  {
    return xposMtx_;
  }
  inline const mtxT&
  xnegMtx(void) const
  {
    return xnegMtx_;
  }
  inline const mtxT&
  wposMtx(void) const
  {
    return wposMtx_;
  }
  inline const mtxT&
  wnegMtx(void) const
  {
    return wnegMtx_;
  }
  inline const mtxT&
  rdMtx(void) const
  {
    return rdMtx_;
  }

protected:

  void
  computeAffinity(
    const mtxT& dMtx,
    const vecI& indVec=vecI::Zero(0));

  void
  computeAffinity(void);

  void
  computeTwoViews(
    const mtxT& mtx);

  void
  removeNoise(void);

private:

};

}

}

#endif // NONMETRIC_TWOVIEWS_H
