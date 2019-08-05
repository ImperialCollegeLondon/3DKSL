#include <ksl/clustering/kmeans.h>

namespace ksl
{

namespace clustering
{

template<typename T>
KMeans<T>::KMeans(void)
{}

template<typename T>
KMeans<T>::KMeans(
  const int& initMethod, const int& nIter):
  params_(initMethod, nIter)
{}

template<typename T>
KMeans<T>::KMeans(
  const KMeansParams<T>& params):
  params_(params)
{}

template<typename T>
KMeans<T>::~KMeans(void)
{}

template<typename T>
void
KMeans<T>::computeClusters(
  const mtxT& dMtx)
{
  if(this->bparams_.nClusters<=0)
  {
    return;
  }

  init(dMtx);
  T prevDist=1.0, totalDist=0.0;
  for(int i=0; i<params_.nIter && prevDist!=totalDist; ++i)
  {
    prevDist=totalDist;
    totalDist=assignClosest(dMtx);
    clustersCentresUpdate(dMtx);
  }
}

template<typename T>
void
KMeans<T>::init(
  const mtxT& dMtx)
{
  clustersCentres_.resize(this->bparams_.nClusters, this->bparams_.nDims);
  switch(params_.initMethod)
  {
    case 0: initRandom(dMtx); break;
    case 1: initRandomPlus(dMtx); break;
    default: clustersCentresUpdate(dMtx); break;
  }
}

template<typename T>
T
KMeans<T>::assignClosest(
  const mtxT& dMtx)
{
  T totalDist=0.0;
  mtxT wMtx;
  utils::pairwiseDistanceEuclidean<T>(dMtx, clustersCentres_, wMtx);
  vecI& clusters(this->clusters_);
#ifdef _OPENMP
  #pragma omp parallel for shared(clusters, wMtx) reduction(+ : totalDist)
#endif
  for(int i=0; i<this->bparams_.nPoints; ++i)
  {
    int indMin;
    totalDist+=wMtx.row(i).minCoeff(&indMin);
    clusters(i)=indMin;
  }
  return totalDist;
}

template<typename T>
void
KMeans<T>::clustersCentresUpdate(
  const mtxT& dMtx)
{
  vecI nPerCluster(this->bparams_.nClusters);
  nPerCluster.setZero();
  clustersCentres_.setZero();
  for(int i=0; i<this->bparams_.nPoints; ++i)
  {
    clustersCentres_.row(this->clusters_(i))+=dMtx.row(i);
    ++nPerCluster(this->clusters_(i));
  }
  for(int i=0; i<this->bparams_.nClusters; ++i)
  {
    if(nPerCluster(i)>0)
    {
      clustersCentres_.row(i)/=nPerCluster(i);
    }
  }
}

template<typename T>
void
KMeans<T>::initRandom(
  const mtxT& dMtx)
{
  vecI indVec(vecI::LinSpaced(this->bparams_.nPoints, 0, this->bparams_.nPoints-1));
  for(int i=0; i<this->bparams_.nClusters; ++i)
  {
    const int ind=utils::rnd<int>(0, this->bparams_.nPoints-1-i);
    clustersCentres_.row(i)=dMtx.row(indVec(ind));
    indVec(ind)=indVec(this->bparams_.nPoints-1-i);
  }
}

template<typename T>
void
KMeans<T>::initRandomPlus(
  const mtxT& dMtx)
{
  clustersCentres_.row(0)=dMtx.row(utils::rnd<int>(0, this->bparams_.nPoints-1));
  if(this->bparams_.nClusters<2)
  {
    return;
  }
  vecT curDist((dMtx.rowwise()-clustersCentres_.row(0)).array().square().rowwise().sum());
  vecT minDist(curDist);
  clustersCentres_.row(1)=dMtx.row(discreteRand(minDist));
  for(int i=2; i<this->bparams_.nClusters; ++i)
  {
    curDist=(dMtx.rowwise()-clustersCentres_.row(i-1)).array().square().rowwise().sum();
    minDist=curDist.cwiseMin(minDist);
    clustersCentres_.row(i)=dMtx.row(discreteRand(minDist));
  }
}

template<typename T>
int
KMeans<T>::discreteRand(
  const vecT& vec)
{
  const T r=vec.sum()*utils::rnd<T>(0.0, 1.0);
  T curSum=vec(0);
  int val=0;
  while(r>=curSum && val<vec.size()-1)
  {
    ++val;
    curSum+=vec(val);
  }
  return val;
}

template class KMeans<double>;
template class KMeans<float>;

}

}
