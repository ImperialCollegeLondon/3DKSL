#include <ksl/clustering/label_propagation.h>
#include <iostream>

namespace ksl
{

namespace clustering
{

template<typename T>
LabelPropagation<T>::LabelPropagation(void)
{}

template<typename T>
LabelPropagation<T>::LabelPropagation(
  const T& alpha, const T& tau, const T& initFact,
  const T& thresh, const T& eps, const T& fracSplit,
  const int& nIter, const int& cutType):
  params_(alpha, tau, initFact, thresh, eps, fracSplit, nIter, cutType)
{
  assert(0.0<params_.alpha && params_.alpha<1.0);
  assert(params_.initFact>1.0);
}

template<typename T>
LabelPropagation<T>::LabelPropagation(
  const LabelPropagationParams<T>& params):
  params_(params)
{
  assert(0.0<params_.alpha && params_.alpha<1.0);
  assert(params_.initFact>1.0);
}

template<typename T>
LabelPropagation<T>::~LabelPropagation(void)
{}

template<typename T>
void
LabelPropagation<T>::init(
  const mtxT& yMtx)
{
  yMtx_=yMtx;
}

template<typename T>
void
LabelPropagation<T>::computeClusters(
  const mtxT &dMtx)
{
  assert(this->affinity_!=nullptr);

  this->affinity_->compute(dMtx);
  Eigen::Map<const mtxT> wMtx(this->affinity_->wMtx().data(), this->bparams_.nPoints, this->bparams_.nPoints);

  //t_.tic();

  mtxT lapMtx;
  laplacian<T>(wMtx, lapMtx);

  /* main iteration loop */
  /*const mtxT yiMtx(yMtx_);
  mtxT yMtx;
  T err=10.0*params_.thresh;
  for(int i=0; i<params_.nIter && err>params_.thresh; ++i)
  {
    yMtx=yMtx_;
    yMtx_=(1.0-params_.alpha)*yiMtx;
    yMtx_.noalias()+=params_.alpha*lapMtx*yMtx;
    err=(yMtx_-yMtx).cwiseAbs().maxCoeff();
  }*/
  yMtx_=(1.0-params_.alpha)*(mtxT::Identity(this->bparams_.nPoints, this->bparams_.nPoints)-params_.alpha*lapMtx).llt().solve(yMtx_);

  int nClusters=yMtx_.cols();
  this->bparams_.nClusters=nClusters;
  vecI uVec(this->bparams_.nClusters);
  uVec.setZero();
  for(int i=0; i<this->bparams_.nPoints; ++i)
  {
    yMtx_.row(i).maxCoeff(&this->clusters_(i));
    //yMtx_.row(i)/=yMtx_.row(i).sum();
    yMtx_.row(i).setZero();
    yMtx_(i, this->clusters_(i))=1.0;
    ++uVec(this->clusters_(i));
  }

  /* search to remove clusters */
  for(int i=0; i<this->bparams_.nClusters; ++i)
  {
    if(uVec(i)==0)
    {
      for(int j=i+1; j<this->bparams_.nClusters; ++j)
      {
        yMtx_.col(j-1).swap(yMtx_.col(j));
        uVec.row(j-1).swap(uVec.row(j));
      }
      this->clusters_=(this->clusters_.array()>i).select(this->clusters_.array()-1, this->clusters_);
      --this->bparams_.nClusters, --i;
    }
  }
  if(this->bparams_.nClusters<nClusters)
  {
    yMtx_.conservativeResize(Eigen::NoChange, this->bparams_.nClusters);
  }

  /* find sub affinity matrices */
  std::vector<vecI> indVec(this->bparams_.nClusters);
  std::vector<mtxT> wcMtx(this->bparams_.nClusters);
  for(int k=0; k<this->bparams_.nClusters; ++k)
  {
    indVec[k].resize(uVec(k));
    wcMtx[k].resize(uVec(k), uVec(k));
    wcMtx[k].diagonal().setZero();
  }
  vecI iVec(this->bparams_.nClusters), jVec(this->bparams_.nClusters);
  iVec.setZero();
  for(int i=0; i<this->bparams_.nPoints; ++i)
  {
    const int k=this->clusters_(i);
    jVec(k)=iVec(k)+1;
    indVec[k](iVec(k))=i;
    for(int j=i+1; j<this->bparams_.nPoints; ++j)
    {
      if(this->clusters_(j)==k)
      {
        wcMtx[k](iVec(k), jVec(k))=wMtx(i, j);
        wcMtx[k](jVec(k), iVec(k))=wMtx(j, i);
        ++jVec(k);
      }
    }
    ++iVec(k);
  }

  /* find second largest eigenvalue of each sub affinity matrix */
  vecT vVec, dVec;
  nClusters=this->bparams_.nClusters;
  int numSplit=params_.fracSplit*this->bparams_.nPoints;
  if(numSplit<4)
  {
    numSplit=4;
  }
  for(int k=0; k<this->bparams_.nClusters; ++k)
  {
    if(uVec(k)>numSplit)
    {
      laplacian<T>(wcMtx[k], lapMtx, dVec);
      Spectra::DenseSymMatProd<T> op(lapMtx);
      Spectra::SymEigsSolver<Spectra::DenseSymMatProd<T> > eig(op, 2, 2*2+1);
      eig.init();
      try
      {
        eig.compute(Spectra::SortRule::LargestAlge, 1000, params_.eps);
        if(eig.info()!=Spectra::CompInfo::Successful)
        {
          std::cout<<"[Label Propagation] eigendecomposition not successful"<<std::endl;
          continue;
        }
      }
      catch(const std::exception& e)
      {
        std::cout<<"[Label Propagation] "<<e.what()<<std::endl;
        continue;
      }
      vVec=eig.eigenvectors().col(1);

      /* split ? */
      const T cutThresh=computeCutThresh(vVec);
      const vecI indThreshVec((vVec.array()<cutThresh).select(vecI::Ones(uVec(k)), 0));
      const int indThreshVecSum=indThreshVec.sum();
      if(indThreshVecSum<numSplit || indThreshVec.size()-indThreshVecSum<numSplit)
      {
        continue;
      }
      const T cutValue=computeCutValue(indThreshVec, wcMtx[k], dVec);
      //std::cout<<"cutValue: "<<cutValue<<"/"<<params_.tau<<std::endl;
      if(cutValue<params_.tau)
      {
        yMtx_.conservativeResize(Eigen::NoChange, nClusters+1);
        yMtx_.col(nClusters).setZero();
        for(int i=0; i<uVec(k); ++i)
        {
          const int ind=indVec[k](i);
          if(indThreshVec(i)==1)
          {
            yMtx_(ind, nClusters)=params_.initFact*yMtx_.row(ind).sum();
            this->clusters_(ind)=nClusters;
          }
          else
          {
            yMtx_.row(ind).col(k).swap(yMtx_.row(ind).col(nClusters));
            yMtx_(ind, k)=params_.initFact*yMtx_.row(ind).sum();
          }
          yMtx_.row(ind)/=yMtx_.row(ind).sum();
        }
        ++nClusters;
      }
    }
    else
    {
      for(int i=0; i<uVec(k); ++i)
      {
        //yMtx_.row(indVec[k](i)).setConstant(1.0/yMtx_.cols());
        yMtx_.row(indVec[k](i)).setZero();
      }
    }
  }
  this->bparams_.nClusters=nClusters;

  //tCluster_=t_.toc();
}

template<typename T>
T
LabelPropagation<T>::computeCutThresh(
  const vecT& vVec) const
{
  switch(params_.cutType)
  {
    case 0: return 0.0;
    case 1: return ksl::utils::median<T>(vVec);
    case 2: return vVec.mean();
    default: return 0.0;
  }
}

template<typename T>
T
LabelPropagation<T>::computeCutValue(
  const vecI& indThreshVec,
  const mtxT& wMtx,
  const vecT& dVec) const
{
  const vecT dThreshVec((indThreshVec.replicate(1, wMtx.cols()).array()==1).select(wMtx, 0.0).colwise().sum());
  const T cut=(indThreshVec.array()!=1).select(dThreshVec, 0.0).sum();

  if(cut==0.0)
  {
    return 0.0;
  }
  return (cut/(indThreshVec.array()==1).select(dVec, 0.0).sum())+
    (cut/(indThreshVec.array()!=1).select(dVec, 0.0).sum());
}

template<typename T>
T
LabelPropagation<T>::computeRayleighQuotient(
  const mtxT& dwMtx,
  const vecT& dVec, 
  const vecT& vVec) const
{
  const T num=vVec.transpose()*dwMtx*vVec;
  if(num==0.0)
  {
    return 0.0;
  }
  const T denum=vVec.transpose()*dVec.asDiagonal()*vVec;
  return num/(denum+1.0e-6);
}

template class LabelPropagation<double>;
template class LabelPropagation<float>;

}

}
