#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <cassert>
#include <Eigen/Core>
#include <ksl/affinity/affinity.h>

namespace ksl
{

namespace clustering
{

template<typename T>
struct ClusteringParams
{

  int nPoints, nDims;
  int nClusters;

  ClusteringParams(void):
    nPoints(0), nDims(0),
    nClusters(0)
  {}
  ClusteringParams(
    const int& nPoints, const int& nDims,
    const int& nClusters):
    nPoints(nPoints), nDims(nDims),
    nClusters(nClusters)
  {}
  ClusteringParams(
    const ClusteringParams& params):
    nPoints(params.nPoints), nDims(params.nDims),
    nClusters(params.nClusters)
  {}

  ClusteringParams<T>&
  operator=(
    const ClusteringParams<T>& params)
  {
    nPoints=params.nPoints;
    nDims=params.nDims;
    nClusters=params.nClusters;
    return *this;
  }

};

template<typename T>
class Clustering
{

public:

  typedef Eigen::Matrix<int, Eigen::Dynamic, 1> vecI;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vecT;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mtxI;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mtxT;

private:

protected:

  ClusteringParams<T> bparams_;

  vecI clusters_;

  ksl::affinity::Affinity<T>* affinity_;

public:

  Clustering(void);
  Clustering(
    const int& nPoints, const int& nDims,
    const int& nClusters);
  Clustering(
    const ClusteringParams<T>& params);
  virtual ~Clustering(void);

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
  inline const int&
  nClusters(void) const
  {
    return bparams_.nClusters;
  }
  inline const vecI&
  clusters(void) const
  {
    return clusters_;
  }
  inline const mtxT&
  wMtx(void) const
  {
    return affinity_->wMtx();
  }

  void
  compute(
    const mtxT& dMtx,
    const int& nClusters=0);
  void
  compute(
    const mtxT& dMtx,
    const vecI& clusters,
    const int& nClusters=0);
  void
  compute(
    const mtxT& dMtx,
    ksl::affinity::Affinity<T>* affinity,
    const int& nClusters=0);
  void
  compute(
    const mtxT& dMtx,
    ksl::affinity::Affinity<T>* affinity,
    const vecI& clusters,
    const int& nClusters=0);

protected:

  virtual void
  computeClusters(
    const mtxT& dMtx)=0;

private:

};

template<typename T>
inline void
laplacian(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& wMtx,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lapMtx)
{
  /* normalized Laplacian L=D^(-0.5)WD^(-0.5) */
  Eigen::Matrix<T, Eigen::Dynamic, 1> dinvVec((wMtx.rowwise().sum().array()+T(1.0e-6)).pow(-0.5));
  lapMtx.noalias()=dinvVec.asDiagonal()*wMtx;
  lapMtx*=dinvVec.asDiagonal();
}
template<typename T>
inline void
laplacian(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& wMtx,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lapMtx,
  Eigen::Matrix<T, Eigen::Dynamic, 1>& dVec)
{
  /* normalized Laplacian L=D^(-0.5)WD^(-0.5) */
  dVec=wMtx.rowwise().sum().array()+T(1.0e-6);
  Eigen::Matrix<T, Eigen::Dynamic, 1> dinvVec(dVec.array().pow(-0.5));
  lapMtx.noalias()=dinvVec.asDiagonal()*wMtx;
  lapMtx*=dinvVec.asDiagonal();
}

}

}

#endif // CLUSTERING_H
