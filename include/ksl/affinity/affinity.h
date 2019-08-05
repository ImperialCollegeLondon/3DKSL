#ifndef AFFINITY_H
#define AFFINITY_H

#include <cassert>
#include <Eigen/Core>

namespace ksl
{

namespace affinity
{

template<typename T>
struct AffinityParams
{

  int nPoints, nDims;

  AffinityParams(void):
    nPoints(0), nDims(0)
  {}
  AffinityParams(
    const int& nPoints, const int& nDims):
    nPoints(nPoints), nDims(nDims)
  {}
  AffinityParams(
    const AffinityParams<T>& params):
    nPoints(params.nPoints), nDims(params.nDims)
  {}

};

template<typename T>
class Affinity
{

public:

  typedef Eigen::Matrix<int, Eigen::Dynamic, 1> vecI;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vecT;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mtxI;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mtxT;

private:

protected:

  AffinityParams<T> bparams_;

  mtxT wMtx_;

public:

  Affinity(void);
  Affinity(
    const int& nPoints, const int& nDims);
  Affinity(
    const AffinityParams<T>& params);
  virtual ~Affinity(void);

  inline const int&
  nPoints(void) const
  {
    return bparams_.nPoints;
  }
  inline const int&
  nDims(void) const
  {
    return bparams_.nDims;
  }
  inline const mtxT&
  wMtx(void) const
  {
    return wMtx_;
  }

  void
  compute(
    const mtxT& dMtx);
  void
  compute(
    const mtxT& dMtx,
    const vecI& indVec);

protected:

  virtual void
  computeAffinity(
    const mtxT& dMtx,
    const vecI& indVec=vecI::Zero(0))=0;

private:

};

}

}

#endif // AFFINITY_H
