#ifndef KMEANS_H
#define KMEANS_H

#include <Eigen/Core>
#include <ksl/clustering/clustering.h>
#include <ksl/utils/dataset.h>
#include <ksl/utils/random_generators.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace ksl
{

namespace clustering
{

template<typename T>
struct KMeansParams
{

  int initMethod, nIter;

  KMeansParams(void):
    initMethod(0), nIter(100)
  {}
  KMeansParams(
    const int& initMethod, const int& nIter):
    initMethod(initMethod), nIter(nIter)
  {}
  KMeansParams(
    const KMeansParams<T>& params):
    initMethod(params.initMethod), nIter(params.nIter)
  {}

  KMeansParams<T>&
  operator=(
    const KMeansParams<T>& params)
  {
    initMethod=params.initMethod;
    nIter=params.nIter;
    return *this;
  }

};

template<typename T>
class KMeans : public Clustering<T>
{

public:

  typedef typename Clustering<T>::vecI vecI;
  typedef typename Clustering<T>::vecT vecT;
  typedef typename Clustering<T>::mtxT mtxT;

private:

protected:

  KMeansParams<T> params_;

  mtxT clustersCentres_;

public:

  KMeans(void);
  KMeans(
    const int& initMethod, const int& nIter);
  KMeans(
    const KMeansParams<T>& params);
  ~KMeans(void);

  inline const int&
  initMethod(void) const
  {
    return params_.initMethod;
  }
  inline const int&
  nIter(void) const
  {
    return params_.nIter;
  }
  inline const mtxT&
  clustersCentres(void) const
  {
    return clustersCentres_;
  }

protected:

  void
  computeClusters(
    const mtxT& dMtx);

  void
  init(
    const mtxT& dMtx);

private:

  inline T
  assignClosest(
    const mtxT& dMtx);

  inline void
  clustersCentresUpdate(
    const mtxT& dMtx);

  inline void
  initRandom(
    const mtxT& dMtx);
  inline void
  initRandomPlus(
    const mtxT& dMtx);

  inline int
  discreteRand(
    const vecT& vec);

};

}

}

#endif // KMEANS_H
