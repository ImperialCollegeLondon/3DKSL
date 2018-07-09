#ifndef SPARSE_SUBSPACE_CLUSTERING_HPP
#define SPARSE_SUBSPACE_CLUSTERING_HPP

#include <assert.h>
#include <Eigen/Core>
#include <float.h>
#include <igl/find.h>
#include <igl/sort.h>
#include <igl/unique.h>
#include <ksl/clustering/hungarian.hpp>
#include <ksl/clustering/random_generators.hpp>
#include <ksl/clustering/spectral_clustering.hpp>

namespace ksl
{

namespace clustering
{

class SparseSubspaceClustering
{

private:

  int nData_;
  int nDims_;

protected:

  float err_[3];
  int nGroups_;
  Eigen::MatrixXf dataProj_;
  Eigen::MatrixXf coeff_;
  Eigen::MatrixXf coeffP_;
  Eigen::MatrixXf symGraph_;
  Eigen::VectorXi groups_;

public:

  SparseSubspaceClustering(void);
  virtual ~SparseSubspaceClustering(void);

  float
  err(
    const int ind) const;
  int
  nGroups(void) const;
  const Eigen::MatrixXf&
  dataProj(void) const;
  const Eigen::MatrixXf&
  coeff(void) const;
  const Eigen::MatrixXf&
  coeffP(void) const;
  const Eigen::MatrixXf&
  symGraph(void) const;
  const Eigen::VectorXi&
  groups(void) const;

  virtual void
  compute(
    const Eigen::MatrixXf &data,
    const float alpha=0,
    const int r=0,
    const int affine=0,
    const int outlier=0,
    const int k=0,
    const float rho=1.0,
    const int nGroups=0,
    const int maxGroups=0,
		const float eigThresh=0.0,
    const int nIter=200,
    const float thr1=5.0e-4,
    const float thr2=5.0e-4);

  virtual float
  bestMap(
    const Eigen::VectorXi &groundTruth);

protected:

  virtual void
  projectData(
    const Eigen::MatrixXf &data,
    const int r=0);
  virtual void
  buildAdjacencyMatrix(
    const int k=0,
    const float rho=1.0);

  virtual void
  admmLasso(
    const int affine=0,
    const float alpha=800.0,
    const int nIter=200,
    const float thr1=5.0e-4,
    const float thr2=5.0e-4);
  virtual void
  admmOutlier(
    const int affine=0,
    const float alpha=20.0,
    const int nIter=150,
    const float thr1=5.0e-4,
    const float thr2=5.0e-4);

private:

  float
  computeLambda(
    const Eigen::MatrixXf &mtx);

  float
  errorLinSys(
    const Eigen::MatrixXf &mtx1,
    const Eigen::MatrixXf &mtx2);

  void
  thrC(
    const float rho=1.0);

};

}

}

#endif // SPARSE_SUBSPACE_CLUSTERING_HPP

