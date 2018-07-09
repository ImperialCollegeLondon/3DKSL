#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <algorithm>
#include <Eigen/Core>
#include <stdlib.h>

namespace ksl
{

namespace clustering
{

template<typename T>
class KMeans
{

private:

  int nData_;
  int nClusters_;
  int nDims_;

protected:

  Eigen::VectorXi clusters_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> clustersCentres_;

public:

  KMeans(void);
  virtual ~KMeans(void);

  const Eigen::VectorXi&
  clusters(void) const
  {
    return clusters_;
  }
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&
  clustersCentres(void) const
  {
    return clustersCentres_;
  }

  virtual void
  compute(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
    const int nClusters,
    const int nIter=100,
    const int initMethod=0,
		const Eigen::VectorXi &initClusters=Eigen::VectorXi::Zero(0));

protected:

  virtual void
  init(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
    const int initMethod=0,
		const Eigen::VectorXi &initClusters=Eigen::VectorXi::Zero(0));

  virtual T
  assignClosest(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &dist);

  virtual void
  updateClustersCentres(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data);

  virtual void
  pairwiseDist(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &dist);

  virtual int
  discreteRand(
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &vec);

private:

};

}

}

#endif // KMEANS_HPP

