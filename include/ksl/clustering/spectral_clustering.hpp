#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include <assert.h>
#include <Eigen/Core>
#include <ksl/clustering/kmeans.hpp>
#include <ksl/clustering/rotation_evaluation.hpp>
#include <Spectra/SymEigsSolver.h>

namespace ksl
{

namespace clustering
{

template<typename T>
class SpectralClustering
{

private:

  RotationEvaluation<T> rotEval_;

protected:

  int nData_;
  int nEigVecs_;
	int nClusters_;
  Eigen::VectorXi clusters_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> laplacianMtx_;
  Eigen::Matrix<T, Eigen::Dynamic, 1> eigVals_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigVecs_;

public:

  SpectralClustering(void);
  virtual ~SpectralClustering(void);

	int
	nClusters(void) const
  {
    return nClusters_;
  }
  const Eigen::VectorXi&
  clusters(void) const
  {
    return clusters_;
  }
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&
  laplacian(void) const
  {
    return laplacianMtx_;
  }
  const Eigen::Matrix<T, Eigen::Dynamic, 1>&
  eigVals(void) const
  {
    return eigVals_;
  }
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&
  eigVecs(void) const
  {
    return eigVecs_;
  }

  virtual int
  compute(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &adjacencyMtx,
    const int numClusters=0,
    const T eigThresh=0.0,
    const int numClustersMax=20,
    const int maxIter=1000,
    const T tol=1e-5);

protected:

  void
  clusterRotate(void);

  virtual void
  computeLaplacian(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &adjacencyMtx,
    const int opt=0);
  virtual int
  computeEigs(
    const int nEigs=1,
    const int convergenceParam=2,
    const int maxIter=1000,
    const T tol=1e-10);

private:

};

}

}

#endif // SPECTRAL_CLUSTERING_HPP

