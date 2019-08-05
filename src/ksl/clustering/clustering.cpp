#include <ksl/clustering/clustering.h>

namespace ksl
{

namespace clustering
{

template<typename T>
Clustering<T>::Clustering(void):
  affinity_(nullptr)
{}

template<typename T>
Clustering<T>::Clustering(
  const int& nPoints, const int& nDims,
  const int& nClusters):
  bparams_(nPoints, nDims, nClusters),
  affinity_(nullptr)
{}

template<typename T>
Clustering<T>::Clustering(
  const ClusteringParams<T>& params):
  bparams_(params),
  affinity_(nullptr)
{}

template<typename T>
Clustering<T>::~Clustering(void)
{}

template<typename T>
void
Clustering<T>::compute(
  const mtxT& dMtx,
  const int& nClusters)
{
  compute(dMtx, nullptr, nClusters);
}

template<typename T>
void
Clustering<T>::compute(
  const mtxT& dMtx,
  const vecI& clusters,
  const int& nClusters)
{
  compute(dMtx, nullptr, clusters, nClusters);
}

template<typename T>
void
Clustering<T>::compute(
  const mtxT& dMtx,
  ksl::affinity::Affinity<T>* affinity,
  const int& nClusters)
{
  bparams_.nPoints=dMtx.rows();
  bparams_.nDims=dMtx.cols();
  if(dMtx.rows()!=clusters_.size())
  {
    clusters_.resize(bparams_.nPoints);
  }
  bparams_.nClusters=nClusters;
  affinity_=affinity;
  this->computeClusters(dMtx);
}

template<typename T>
void
Clustering<T>::compute(
  const mtxT& dMtx,
  ksl::affinity::Affinity<T>* affinity,
  const vecI& clusters,
  const int& nClusters)
{
  assert(dMtx.rows()==clusters.size());

  clusters_=clusters;
  (nClusters<=0) ? compute(dMtx, affinity, clusters_.maxCoeff()) : compute(dMtx, affinity, nClusters);
}

template class Clustering<double>;
template class Clustering<float>;

}

}
