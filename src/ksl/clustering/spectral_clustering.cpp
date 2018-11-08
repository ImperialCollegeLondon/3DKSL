#include <ksl/clustering/spectral_clustering.hpp>
#include <iostream>

namespace ksl
{

namespace clustering
{

template<typename T>
SpectralClustering<T>::SpectralClustering(void)
{}

template<typename T>
SpectralClustering<T>::~SpectralClustering(void)
{}

template<typename T> int
SpectralClustering<T>::compute(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &adjacencyMtx,
  const int nClusters,
  const T eigThresh,
  const int numClustersMax,
  const int maxIter,
  const T tol)
{
  /* adjacencyMtx must be real symmetric */
  assert(adjacencyMtx.rows()==adjacencyMtx.cols());

  nData_=adjacencyMtx.rows();
  if(nClusters<=0)
  {
    /* maximum number of clusters to be considered */
    (nData_>=40) ? nEigVecs_=numClustersMax : nEigVecs_=nData_*0.5;
    nClusters_=0;
  }
	else
	{
    nEigVecs_=nClusters;
    nClusters_=nClusters;
    if(nClusters_==1)
    {
      clusters_.resize(nData_);
      clusters_.setZero();
      return nClusters_;
    }
	}

  /* normalized Laplacian L=I-D^(-1/2)(D-A)D^(-1/2) */
  computeLaplacian(adjacencyMtx, 1);

  int ncEigs=computeEigs(nEigVecs_, 2, maxIter, tol);
  if(ncEigs<nEigVecs_)
  {
    return ncEigs;
  }
  
  /* cluster by self-tunig */
  if(nClusters==0)
  {
    std::cout<<eigVals_.transpose()<<std::endl;
    while(eigVals_(nEigVecs_-1)<eigThresh && nEigVecs_>1)
    {
      nEigVecs_--;
    }
    if(nEigVecs_==1)
    {
      nClusters_=1;
      clusters_.resize(nData_);
      clusters_.setZero();
      return nClusters_;
    }

    clusterRotate();
  }
  /* cluster by kmeans */
  else
  {
    KMeans<T> km; 
    km.compute(eigVecs_, nClusters_, 100, 1);
    clusters_=km.clusters();
  }

  return nClusters_;
}

template<typename T> void
SpectralClustering<T>::clusterRotate(void)
{
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigVecsRotated;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigVecs(eigVecs_.leftCols(2));
  rotEval_.compute(eigVecs);
  T maxRotationQuality=rotEval_.rotationQuality();
  clusters_=rotEval_.clusters();
  eigVecsRotated=rotEval_.eigVecsRotated();

	nClusters_=2;
  std::cout<<"2/"<<nEigVecs_<<" eval: "<<rotEval_.rotationQuality()<<std::endl;
  for(int i=3; i<=nEigVecs_; i++)
  {
		eigVecs=rotEval_.eigVecsRotated();
    eigVecs.conservativeResize(Eigen::NoChange, i);
    eigVecs.col(i-1)=eigVecs_.col(i-1);

    rotEval_.compute(eigVecs);
    std::cout<<i<<"/"<<nEigVecs_<<" eval: "<<rotEval_.rotationQuality()<<std::endl;
    if((rotEval_.rotationQuality()-maxRotationQuality>-2.5e-3) || rotEval_.rotationQuality()>=1.0)
    {
      maxRotationQuality=rotEval_.rotationQuality();
      clusters_=rotEval_.clusters();
      eigVecsRotated=rotEval_.eigVecsRotated();
			nClusters_=i;
    }
  }

	/* last kmeans clustering step */
	KMeans<T> km;
  km.compute(eigVecsRotated, nClusters_, 100, 2, clusters_);
  clusters_=km.clusters();
}

template<typename T> void
SpectralClustering<T>::computeLaplacian(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &adjacencyMtx,
  const int opt)
{
  /* adjacencyMtx must be real symmetric */
  assert(adjacencyMtx.rows()==adjacencyMtx.cols());

  /* degree of adjacency matrix */
  Eigen::Matrix<T, Eigen::Dynamic, 1> degreeMtx=adjacencyMtx.rowwise().sum();

  /* non-normalized laplacian L=D-A */
  laplacianMtx_=degreeMtx.asDiagonal();
  laplacianMtx_-=adjacencyMtx;
  /* normalized laplacian L=I-D^(-1/2)(D-A)D^(-1/2) */
  if(opt==1)
  {
    Eigen::Matrix<T, Eigen::Dynamic, 1> degreeSqrtInvMtx;
    degreeSqrtInvMtx=(degreeMtx.array()+T(1e-16)).sqrt().pow(-1);
    laplacianMtx_.noalias()=degreeSqrtInvMtx.asDiagonal()*laplacianMtx_;
    laplacianMtx_*=degreeSqrtInvMtx.asDiagonal();
    laplacianMtx_=Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(nData_, nData_)-laplacianMtx_;
  }
}

template<typename T> int
SpectralClustering<T>::computeEigs(
  const int nEigs,
  const int convergenceParam,
  const int maxIter,
  const T tol)
{
  Spectra::DenseSymMatProd<T> op(laplacianMtx_);
  Spectra::SymEigsSolver<T, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<T> > eigs(&op, nEigs, convergenceParam*nEigs);

  eigs.init();
  int ncEigs=eigs.compute(maxIter, tol);
  if(eigs.info()!=Spectra::SUCCESSFUL)
  {
    if(eigs.info()==Spectra::NOT_COMPUTED)
    {
      return -1;
    }
    else if(eigs.info()==Spectra::NOT_CONVERGING)
    {
      return -2;
    }
    else if(eigs.info()==Spectra::NUMERICAL_ISSUE)
    {
      return -3;
    }
    return -4;
  }
  eigVals_=eigs.eigenvalues();
  eigVecs_=eigs.eigenvectors();
  return ncEigs;
}

template class SpectralClustering<double>;
template class SpectralClustering<float>;

}

}

