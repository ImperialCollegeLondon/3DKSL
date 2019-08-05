#ifndef LABEL_PROPAGATION_H
#define LABEL_PROPAGATION_H

#include <cassert>
#include <Eigen/Dense>
#include <ksl/clustering/clustering.h>
#include <ksl/utils/dataset.h>
//#include <ksl/utils/tictoc.h>
#include <Spectra/SymEigsSolver.h>
#include <vector>

namespace ksl
{

namespace clustering
{

template<typename T>
struct LabelPropagationParams
{

  T alpha, tau, initFact;
  T thresh, eps, fracSplit;
  int nIter, cutType;

  LabelPropagationParams(void):
    alpha(0.1), tau(1.0e-3), initFact(1.5),
    thresh(1.0e-6), eps(1.0e-6), fracSplit(0.0),
    nIter(200), cutType(0)
  {}
  LabelPropagationParams(
    const T& alpha, const T& tau, const T& initFact,
    const T& thresh, const T& eps, const T& fracSplit,
    const int& nIter, const int& cutType):
    alpha(alpha), tau(tau), initFact(initFact),
    thresh(thresh), eps(eps), fracSplit(fracSplit),
    nIter(nIter), cutType(cutType)
  {}
  LabelPropagationParams(
    const LabelPropagationParams<T>& params):
    alpha(params.alpha), tau(params.tau), initFact(params.initFact),
    thresh(params.thresh), eps(params.eps), fracSplit(params.fracSplit),
    nIter(params.nIter), cutType(params.cutType)
  {}

  LabelPropagationParams<T>&
  operator=(
    const LabelPropagationParams<T>& params)
  {
    alpha=params.alpha;
    tau=params.tau;
    initFact=params.initFact;
    thresh=params.thresh;
    eps=params.eps;
    fracSplit=params.fracSplit;
    nIter=params.nIter;
    cutType=params.cutType;
    return *this;
  }

};

template<typename T>
class LabelPropagation : public Clustering<T>
{

public:

  typedef typename Clustering<T>::vecI vecI;
  typedef typename Clustering<T>::vecT vecT;
  typedef typename Clustering<T>::mtxT mtxT;

private:

  //utils::TicToc t_;
  //double tCluster_;

protected:

  LabelPropagationParams<T> params_;

  mtxT yMtx_;

public:

  LabelPropagation(void);
  LabelPropagation(
    const T& alpha, const T& tau, const T& initFact,
    const T& thresh, const T& eps, const T& fracSplit,
    const int& nIter, const int& cutType);
  LabelPropagation(
    const LabelPropagationParams<T>& params);
  ~LabelPropagation(void);

  /*inline const double&
  tCluster(void) const
  {
    return tCluster_;
  }*/

  inline const T&
  alpha(void) const
  {
    return params_.alpha;
  }
  inline const T&
  tau(void) const
  {
    return params_.tau;
  }
  inline const T&
  initFact(void) const
  {
    return params_.initFact;
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
  inline const T&
  fracSplit(void) const
  {
    return params_.fracSplit;
  }
  inline const int&
  nIter(void) const
  {
    return params_.nIter;
  }
  inline const int&
  cutType(void) const
  {
    return params_.cutType;
  }
  inline const mtxT&
  yMtx(void) const
  {
    return yMtx_;
  }
  inline mtxT&
  yMtx(void)
  {
    return yMtx_;
  }

  void
  init(
    const mtxT& yMtx);

protected:

  void
  computeClusters(
    const mtxT& dMtx);

  inline T
  computeCutThresh(
    const vecT& vVec) const;
  inline T
  computeCutValue(
    const vecI& indThreshVec,
    const mtxT& wMtx,
    const vecT& dVec) const;

  inline T
  computeRayleighQuotient(
    const mtxT& dwMtx,
    const vecT& dVec, 
    const vecT& vVec) const;

private:

};

}

}

#endif // LABEL_PROPAGATION_H
