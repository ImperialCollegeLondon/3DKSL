#ifndef SPARSE_SUBSPACE_H
#define SPARSE_SUBSPACE_H

#include <cmath>
#include <Eigen/Dense>
#include <igl/sort.h>
#include <ksl/affinity/affinity.h>

namespace ksl
{

namespace affinity
{

template<typename T>
struct SparseSubspaceParams
{

  T alpha, rho;
  int k, r, nIter;
  bool affine, outlier;
  T thresh1, thresh2;

  SparseSubspaceParams(void):
    alpha(0.0), rho(1.0),
    k(0), r(0), nIter(200),
    affine(false), outlier(false),
    thresh1(5.0e-4), thresh2(5.0e-4)
  {}
  SparseSubspaceParams(
    const T& alpha, const T& rho,
    const int& k, const int& r, const int& nIter,
    const bool& affine, const bool& outlier,
    const T& thresh1, const T& thresh2):
    alpha(alpha), rho(rho),
    k(k), r(r), nIter(nIter),
    affine(affine), outlier(outlier),
    thresh1(thresh1), thresh2(thresh2)
  {}
  SparseSubspaceParams(
    const SparseSubspaceParams& params):
    alpha(params.alpha), rho(params.rho),
    k(params.k), r(params.r), nIter(params.nIter),
    affine(params.affine), outlier(params.outlier),
    thresh1(params.thresh1), thresh2(params.thresh2)
  {}

  SparseSubspaceParams<T>&
  operator=(
    const SparseSubspaceParams<T>& params)
  {
    alpha=params.alpha;
    rho=params.rho;
    k=params.k;
    r=params.r;
    nIter=params.nIter;
    affine=params.affine;
    outlier=params.outlier;
    thresh1=params.thresh1;
    thresh2=params.thresh2;
    return *this;
  }

};

template<typename T>
class SparseSubspace : public Affinity<T>
{

public:

  typedef typename Affinity<T>::vecI vecI;
  typedef typename Affinity<T>::vecT vecT;
  typedef typename Affinity<T>::mtxI mtxI;
  typedef typename Affinity<T>::mtxT mtxT;

private:

protected:

  SparseSubspaceParams<T> params_;

  vecT deltaVec_;
  mtxT cMtx_, lambdaMtx_;
  mtxT dMtx_;

public:

  SparseSubspace(void);
  SparseSubspace(
    const T& alpha, const T& rho,
    const int& k, const int& r, const int& nIter,
    const bool& affine, const bool& outlier,
    const T& thresh1, const T& thresh2);
  SparseSubspace(
    const SparseSubspaceParams<T>& params);
  ~SparseSubspace(void);

  inline const T&
  alpha(void) const
  {
    return params_.alpha;
  }
  inline const T&
  rho(void) const
  {
    return params_.rho;
  }
  inline const int&
  k(void) const
  {
    return params_.k;
  }
  inline const int&
  r(void) const
  {
    return params_.r;
  }
  inline const int&
  nIter(void) const
  {
    return params_.nIter;
  }
  inline const bool&
  affine(void) const
  {
    return params_.affine;
  }
  inline const bool&
  outlier(void) const
  {
    return params_.outlier;
  }
  inline const T&
  thresh1(void) const
  {
    return params_.thresh1;
  }
  inline const T&
  thresh2(void) const
  {
    return params_.thresh2;
  }

protected:

  void
  computeAffinity(
    const mtxT& dMtx,
    const vecI& indVec=vecI::Zero(0));

  inline void
  admmLasso(void);
  inline void
  admmOutlier(void);

  inline void
  computeAffinity(void);
  inline void
  buildAffinity(void);

  inline void
  project(
    const mtxT& dMtx);

private:

  inline T
  computeLambda(
    const mtxT& mtx) const;

  inline T
  errorCoeff(
    const mtxT& mtx) const;
  inline T
  errorLinSys(
    const mtxT& mtx1,
    const mtxT& mtx2) const;

  inline void
  resize(
    const int& n);

};

}

}

#endif // SPARSE_SUBSPACE_H
