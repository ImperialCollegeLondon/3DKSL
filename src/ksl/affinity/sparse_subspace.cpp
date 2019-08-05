#include <ksl/affinity/sparse_subspace.h>

namespace ksl
{

namespace affinity
{

template<typename T>
SparseSubspace<T>::SparseSubspace(void)
{}

template<typename T>
SparseSubspace<T>::SparseSubspace(
  const T& alpha, const T& rho,
  const int& k, const int& r, const int& nIter,
  const bool& affine, const bool& outlier,
  const T& thresh1, const T& thresh2):
  params_(alpha, rho, k, r, nIter, affine, outlier, thresh1, thresh2)
{}

template<typename T>
SparseSubspace<T>::SparseSubspace(
  const SparseSubspaceParams<T>& params):
  params_(params)
{}

template<typename T>
SparseSubspace<T>::~SparseSubspace(void)
{}

template<typename T>
void
SparseSubspace<T>::computeAffinity(
  const mtxT& dMtx,
  const vecI& indVec)
{
  if(indVec.size()!=0)
  {
    assert(dMtx.rows()==indVec.size());

    int sizeDiff;
    if(this->bparams_.nPoints<dMtx.rows())
    {
      sizeDiff=dMtx.rows()-this->bparams_.nPoints;

      resize(dMtx.rows());

      deltaVec_.tail(sizeDiff).setZero();
      cMtx_.topRightCorner(this->bparams_.nPoints, sizeDiff).setZero();
      cMtx_.bottomRows(sizeDiff).setZero();
      lambdaMtx_.topRightCorner(this->bparams_.nPoints, sizeDiff).setZero();
      lambdaMtx_.bottomRows(sizeDiff).setZero();
    }
    vecI indaVec(indVec);
    for(int i=0; i<indaVec.size(); ++i)
    {
      if(i!=indaVec(i))
      {
        deltaVec_.row(i).swap(deltaVec_.row(indaVec(i)));
        cMtx_.row(i).swap(cMtx_.row(indaVec(i))), cMtx_.col(i).swap(cMtx_.col(indaVec(i)));
        lambdaMtx_.row(i).swap(lambdaMtx_.row(indaVec(i))), lambdaMtx_.col(i).swap(lambdaMtx_.col(indaVec(i)));
        indaVec.row(i).swap(indaVec.row(indaVec(i)));
      }
    }
    if(this->bparams_.nPoints>dMtx.rows())
    {
      resize(dMtx.rows());
    }
  }
  this->bparams_.nPoints=dMtx.rows(), this->bparams_.nDims=dMtx.cols();
  if(indVec.size()==0)
  {
    deltaVec_.setZero(this->bparams_.nPoints);
    cMtx_.setZero(this->bparams_.nPoints, this->bparams_.nPoints);
    lambdaMtx_.setZero(this->bparams_.nPoints, this->bparams_.nPoints);
  }

  project(dMtx.transpose());
  computeAffinity();
}

template<typename T>
void
SparseSubspace<T>::admmLasso(void)
{
  const T alpha=(params_.alpha==0.0) ? 800.0 : params_.alpha;
  const T mu1=alpha/computeLambda(dMtx_), mu2=alpha, mu3=1.0/mu2;

  T err1=10.0*params_.thresh1;
  mtxT aMtx, cMtx, errMtx;
  mtxT mu1ddMtx;
  mu1ddMtx.noalias()=mu1*dMtx_.transpose()*dMtx_;

  mtxT invMtx(mu1ddMtx);
  invMtx.diagonal().array()+=mu2;

  if(!params_.affine)
  {
    invMtx=invMtx.inverse();

    for(int i=0; i<params_.nIter && err1>params_.thresh1; ++i)
    {
      // updating A
      aMtx=invMtx;
      aMtx*=mu1ddMtx+mu2*cMtx_-lambdaMtx_;
      aMtx.diagonal().setZero();
      // updating C
      cMtx=aMtx+mu3*lambdaMtx_;
      cMtx_=cMtx.array().abs()-mu3;
      cMtx_=cMtx_.cwiseMax(0.0).cwiseProduct(cMtx.cwiseSign());
      cMtx_.diagonal().setZero();
      // updating Lagrange multipliers
      errMtx=aMtx-cMtx_;
      lambdaMtx_+=mu2*errMtx;
      //updating errors
      err1=errorCoeff(errMtx);
      //std::cout<<err1<<" "<<i<<"/"<<params_.nIter<<std::endl;
    }
  }
  else
  {
    T err3=10.0*params_.thresh1;
    vecT errVec;
    vecT onesVec(this->bparams_.nPoints);
    onesVec.setOnes();

    invMtx.array()+=mu2;
    invMtx=invMtx.inverse();

    for(int i=0; i<params_.nIter && (err1>params_.thresh1 || err3>params_.thresh1); ++i)
    {
      // updating A
      aMtx=invMtx;
      aMtx*=mu1ddMtx+mu2*(cMtx_+(onesVec*(onesVec.transpose()-mu3*deltaVec_.transpose())))-lambdaMtx_;
      aMtx.diagonal().setZero();
      // updating C
      cMtx=aMtx+mu3*lambdaMtx_;
      cMtx_=cMtx.array().abs()-mu3;
      cMtx_=cMtx_.cwiseMax(0.0).cwiseProduct(cMtx.cwiseSign());
      cMtx_.diagonal().setZero();
      // updating Lagrange multipliers
      errMtx=aMtx-cMtx_;
      lambdaMtx_+=mu2*errMtx;
      errVec.noalias()=aMtx.transpose()*onesVec-onesVec;
      deltaVec_+=mu2*errVec;
      // updating errors
      err1=errorCoeff(errMtx);
      err3=errorCoeff(errVec);
      //std::cout<<err1<<" "<<err3<<" "<<i<<"/"<<params_.nIter<<std::endl;
    }
  }
}

template<typename T>
void
SparseSubspace<T>::admmOutlier(void)
{
  const T alpha=(params_.alpha==0.0) ? 20.0 : params_.alpha;
  mtxT pMtx(this->bparams_.nDims, this->bparams_.nPoints+this->bparams_.nDims);
  pMtx<<dMtx_, vecT::Constant(this->bparams_.nDims, dMtx_.cwiseAbs().colwise().sum().maxCoeff()/1.0).asDiagonal();
  const T mu1=alpha/computeLambda(pMtx), mu2=alpha, mu3=1.0/mu2, mu4=1.0/mu1;

  T err1=10.0*params_.thresh1, err2=10.0*params_.thresh2;
  mtxT cMtx(this->bparams_.nPoints+this->bparams_.nDims, this->bparams_.nPoints), c1Mtx;
  cMtx.setZero();
  mtxT lambda1Mtx(this->bparams_.nDims, this->bparams_.nPoints);
  mtxT lambda2Mtx(this->bparams_.nPoints+this->bparams_.nDims, this->bparams_.nPoints);
  lambda1Mtx.setZero(), lambda2Mtx.setZero();
  mtxT aMtx, errMtx;
  mtxT mu1pMtx(mu1*pMtx);

  mtxT invMtx;
  invMtx.noalias()=mu1pMtx.transpose()*pMtx;
  invMtx.diagonal().array()+=mu2;

  if(!params_.affine)
  {
    invMtx=invMtx.inverse();

    for(int i=0; i<params_.nIter && (err1>params_.thresh1 || err2>params_.thresh2); ++i)
    {
      // updating A
      aMtx=invMtx;
      aMtx*=mu1pMtx.transpose()*(dMtx_+mu4*lambda1Mtx)+mu2*cMtx-lambda2Mtx;
      aMtx.topLeftCorner(this->bparams_.nPoints, aMtx.cols()).diagonal().setZero();
      // updating C
      c1Mtx=aMtx+mu3*lambda2Mtx;
      cMtx=c1Mtx.array().abs()-mu3;
      cMtx=cMtx.cwiseMax(0.0).cwiseProduct(c1Mtx.cwiseSign());
      cMtx.topLeftCorner(this->bparams_.nPoints, cMtx.cols()).diagonal().setZero();
      // updating Lagrange multipliers
      errMtx=aMtx-cMtx;
      lambda1Mtx+=mu1*(dMtx_-pMtx*aMtx);
      lambda2Mtx+=mu2*errMtx;
      // updating errors
      err1=errorCoeff(errMtx);
      err2=errorLinSys(pMtx, aMtx);
      //std::cout<<err1<<" "<<err2<<" "<<i<<"/"<<params_.nIter<<std::endl;
    }
  }
  else
  {
    T err3=10.0*params_.thresh1;
    vecT errVec;
    vecT deltaVec(this->bparams_.nPoints+this->bparams_.nDims);
    deltaVec.head(this->bparams_.nPoints).setOnes(), deltaVec.tail(this->bparams_.nDims).setZero();
    vecT lambdaVec(this->bparams_.nPoints);
    lambdaVec.setZero();
    vecT onesVec(this->bparams_.nPoints);
    onesVec.setOnes();

    invMtx.topLeftCorner(this->bparams_.nPoints, this->bparams_.nPoints).array()+=mu2;
    invMtx=invMtx.inverse();

    for(int i=0; i<params_.nIter && (err1>params_.thresh1 || err2>params_.thresh2 || err3>params_.thresh1); ++i)
    {
      // updating A
      aMtx=invMtx;
      aMtx*=mu1pMtx.transpose()*(dMtx_+mu4*lambda1Mtx)+mu2*(cMtx+deltaVec*(onesVec.transpose()-mu3*lambdaVec.transpose()))-lambda2Mtx;
      aMtx.topLeftCorner(this->bparams_.nPoints, aMtx.cols()).diagonal().setZero();
      // updating C
      c1Mtx=aMtx+mu3*lambda2Mtx;
      cMtx=c1Mtx.array().abs()-mu3;
      cMtx=cMtx.cwiseMax(0.0).cwiseProduct(c1Mtx.cwiseSign());
      cMtx.topLeftCorner(this->bparams_.nPoints, cMtx.cols()).diagonal().setZero();
      // updating Lagrange multipliers
      errMtx=aMtx-cMtx;
      lambda1Mtx+=mu1*(dMtx_-pMtx*aMtx);
      lambda2Mtx+=mu2*errMtx;
      errVec.noalias()=deltaVec.transpose()*aMtx-onesVec.transpose();
      lambdaVec+=mu2*errVec;
      // updating errors
      err1=errorCoeff(errMtx);
      err2=errorLinSys(pMtx, aMtx);
      err3=errorCoeff(errVec);
      //std::cout<<err1<<" "<<err2<<" "<<err2<<" "<<i<<"/"<<params_.nIter<<std::endl;
    }
  }
  cMtx_=cMtx.topRows(this->bparams_.nPoints);
}

template<typename T>
void
SparseSubspace<T>::computeAffinity(void)
{
  (!params_.outlier) ? admmLasso() : admmOutlier();
  buildAffinity();
}

template<typename T>
void
SparseSubspace<T>::buildAffinity(void)
{
  mtxT sMtx(this->bparams_.nPoints, this->bparams_.nPoints);
  mtxI indMtx(this->bparams_.nPoints, this->bparams_.nPoints);
  igl::sort(cMtx_.cwiseAbs(), 1, false, sMtx, indMtx);

  if(params_.rho<1.0)
  {
    mtxT cMtx(cMtx_);
    cMtx_.setZero();
    for(int i=0; i<this->bparams_.nPoints; ++i)
    {
      const T c=sMtx.col(i).sum();
      T cSum=0.0;
      for(int j=0; cSum<params_.rho*c; ++j)
      {
        cSum+=sMtx(j, i);
        cMtx_(indMtx(j, i), i)=cMtx(indMtx(j, i), i);
      }
    }
  }

  mtxT cabsMtx(cMtx_.cwiseAbs());
  if(params_.k==0)
  {
    for(int i=0; i<this->bparams_.nPoints; ++i)
    {
      cabsMtx.col(i)/=cabsMtx(indMtx(0, i), i)+1.0e-6;
    }
  }
  else
  {
    for(int i=0; i<this->bparams_.nPoints; ++i)
    {
      for(int j=0; j<params_.k; ++j)
      {
        cabsMtx(indMtx(j, i), i)/=cabsMtx(indMtx(0, i), i)+1.0e-6;
      }
    }
  }
  this->wMtx_.noalias()=0.5*(cabsMtx+cabsMtx.transpose());
}

template<typename T>
void
SparseSubspace<T>::project(
  const mtxT& dMtx)
{
  if(params_.r==0)
  {
    dMtx_=dMtx;
  }
  else
  {
    Eigen::JacobiSVD<mtxT> svd(dMtx, Eigen::ComputeThinU | Eigen::ComputeThinV);
    dMtx_.noalias()=svd.matrixU().topLeftCorner(this->bparams_.nDims, params_.r).transpose()*dMtx;
  }
}

template<typename T>
T
SparseSubspace<T>::computeLambda(
  const mtxT& mtx) const
{
  mtxT aMtx((mtx.transpose()*dMtx_).topRows(this->bparams_.nPoints));
  aMtx.diagonal().setZero();
  return aMtx.cwiseAbs().colwise().maxCoeff().minCoeff()+1.0e-6;
}

template<typename T>
T
SparseSubspace<T>::errorCoeff(
  const mtxT& mtx) const
{
  return mtx.cwiseAbs().maxCoeff();
}

template<typename T>
T
SparseSubspace<T>::errorLinSys(
  const mtxT& mtx1,
  const mtxT& mtx2) const
{
  const int rows1=mtx1.rows(), cols1=mtx1.cols();
  const int rows2=mtx2.rows(), cols2=mtx2.cols();
  mtxT yMtx, y0Mtx, cMtx;

  if(rows2>cols2)
  {
    const mtxT eMtx(mtx1.block(0, cols2, rows1, cols1-cols2)*mtx2.block(cols2, 0, rows2-cols2, cols2));
    yMtx=mtx1.topLeftCorner(rows1, cols2);
    y0Mtx=yMtx-eMtx;
    cMtx=mtx2.topLeftCorner(cols2, cols2);
  }
  else
  {
    yMtx=mtx1;
    y0Mtx=mtx1;
    cMtx=mtx2;
  }

  // Y0 matrix normalization
  mtxT ynMtx(y0Mtx.rows(), cols2);
  vecT nVec(cols2);
  for(int i=0; i<cols2; ++i)
  {
    nVec(i)=y0Mtx.col(i).norm();
    ynMtx.col(i)=y0Mtx.col(i)/nVec(i);
  }

  const mtxT mMtx(nVec.transpose().replicate(yMtx.rows(), 1));
  const mtxT sMtx(ynMtx-(yMtx*cMtx).cwiseQuotient(mMtx));
  return std::sqrt(sMtx.array().square().colwise().sum().maxCoeff());
}

template<typename T>
void
SparseSubspace<T>::resize(
  const int& n)
{
  deltaVec_.conservativeResize(n);
  cMtx_.conservativeResize(n, n);
  lambdaMtx_.conservativeResize(n, n);
}

template class SparseSubspace<double>;
template class SparseSubspace<float>;

}

}

