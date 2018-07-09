#include <ksl/clustering/kmeans.hpp>

namespace ksl
{

namespace clustering
{

template<typename T>
KMeans<T>::KMeans(void)
{}

template<typename T>
KMeans<T>::~KMeans(void)
{}

template<typename T> void
KMeans<T>::compute(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
  const int nClusters,
  const int nIter,
  const int initMethod,
	const Eigen::VectorXi &initClusters)
{
  nData_=data.rows();
  nClusters_=nClusters;
  nDims_=data.cols();

  init(data, initMethod, initClusters);

  T prevDist=1.0, totalDist=0.0;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dist(nData_, nClusters_);
  dist.setZero();
  for(int i=0; i<nIter && prevDist!=totalDist; i++)
  {
    prevDist=totalDist;
    totalDist=assignClosest(data, dist);
    updateClustersCentres(data);
  }
}

template<typename T> void
KMeans<T>::init(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
  const int initMethod,
	const Eigen::VectorXi &initClusters)
{
  clusters_.resize(nData_);
  clusters_.setZero();
  clustersCentres_.resize(nClusters_, nDims_);

  /* random sample from data rows */
  if(initMethod==0)
  {
    Eigen::VectorXi ind(Eigen::VectorXi::LinSpaced(nData_, 0, nData_-1));
    std::random_shuffle(ind.data(), ind.data()+nData_);
    for(int i=0; i<nClusters_; i++)
    {
      clustersCentres_.row(i)=data.row(ind(i));
    }
  }
  /* plusplus random sample from data rows */
  else if(initMethod==1)
  {
    int choice=rand()%nData_;
    clustersCentres_.row(0)=data.row(choice);
    Eigen::Matrix<T, Eigen::Dynamic, 1> curDist((data.rowwise()-clustersCentres_.row(0)).array().square().rowwise().sum());
    Eigen::Matrix<T, Eigen::Dynamic, 1> minDist(curDist);

    if(nClusters_<2)
    {
      return;
    }

    choice=discreteRand(minDist);
    clustersCentres_.row(1)=data.row(choice);
    for(int i=2; i<nClusters_; i++)
    {
      curDist=(data.rowwise()-clustersCentres_.row(i-1)).array().square().rowwise().sum();
      minDist=curDist.cwiseMin(minDist);
      choice=discreteRand(minDist);
      clustersCentres_.row(i)=data.row(choice);
    }
  }
  /* given prior clusters */
	else if(initMethod==2)
	{
		clusters_=initClusters;
		updateClustersCentres(data);
	}
}

template<typename T> T
KMeans<T>::assignClosest(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &dist)
{
  T totalDist=0.0;
  int indMin;

  pairwiseDist(data, dist);

  for(int i=0; i<nData_; i++)
  {
    totalDist+=dist.row(i).minCoeff(&indMin);
    clusters_(i)=indMin;
  }
  return totalDist;
}

template<typename T> void
KMeans<T>::updateClustersCentres(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> nPerCluster(nClusters_);
  nPerCluster.setZero();
  clustersCentres_.setZero();

  for(int i=0; i<nData_; i++)
  {
    clustersCentres_.row(clusters_(i))+=data.row(i);
    nPerCluster(clusters_(i))++;
  }
  for(int i=0; i<nClusters_; i++)
  {
    if(nPerCluster(i)>0.0)
    {
      clustersCentres_.row(i)/=nPerCluster(i);
    }
  }
}

template<typename T> void
KMeans<T>::pairwiseDist(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &dist)
{
  if(nDims_<16)
  {
    for(int i=0; i<nClusters_; i++)
    {
      dist.col(i)=(data.rowwise()-clustersCentres_.row(i)).array().square().rowwise().sum();
    }
  }
  else
  {
    dist.noalias()=-2.0*data*clustersCentres_.transpose();
    dist.array().rowwise()+=clustersCentres_.array().square().rowwise().sum().transpose().row(0);
  }
}

template<typename T> int
KMeans<T>::discreteRand(
  const Eigen::Matrix<T, Eigen::Dynamic, 1> &vec)
{
  T r=vec.sum()*rand()/RAND_MAX;
  T curSum=vec(0);
  int val=0;

  while(r>=curSum && val<vec.size()-1)
  {
    val++;
    curSum+=vec(val);
  }

  return val;
}

template class KMeans<double>;
template class KMeans<float>;

}

}

