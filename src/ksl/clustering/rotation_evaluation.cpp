#include <ksl/clustering/rotation_evaluation.hpp>

namespace ksl
{

namespace clustering
{

template<typename T>
RotationEvaluation<T>::RotationEvaluation(void)
{}

template<typename T>
RotationEvaluation<T>::~RotationEvaluation(void)
{}

template<typename T> void
RotationEvaluation<T>::compute(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
  const int nIter)
{
  nData_=eigVecs.rows();
  nDims_=eigVecs.cols();
  nAngles_=nDims_*(nDims_-1)*0.5;
  ik_.resize(nAngles_);
  jk_.resize(nAngles_);
  clusters_.resize(nData_);

  /* index mapping (upper triangle) */
  int k=0;
  for(int i=0; i<nDims_-1; i++)
  {
    for(int j=i+1; j<nDims_; j++)
    {
      ik_(k)=i;
      jk_(k)=j;
      k++;
    }
  }

  evaluateRotation(eigVecs, nIter);
}

template<typename T> void
RotationEvaluation<T>::evaluateRotation(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
  const int nIter)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> curTheta(nAngles_);
  Eigen::Matrix<T, Eigen::Dynamic, 1> newTheta(nAngles_);
  curTheta.setZero();
  newTheta.setZero();

  /* initial quality */
  T curRotationQuality=evaluateQuality(eigVecs), newRotationQuality;
  T dRotationQuality;
  T rotationQuality1=curRotationQuality;
  T rotationQuality2=curRotationQuality;

  /* quality refinement */
  for(int i=0; i<nIter; i++)
  {
    for(int j=0; j<nAngles_; j++)
    {
      dRotationQuality=evaluateQualityGradient(eigVecs, curTheta, j);
      newTheta(j)=curTheta(j)-dRotationQuality;
      rotateEigenVectors(eigVecs, newTheta);
      newRotationQuality=evaluateQuality(eigVecsRotated_);

      if(newRotationQuality>curRotationQuality)
      {
        curTheta(j)=newTheta(j);
        curRotationQuality=newRotationQuality;
      }
      else
      {
        newTheta(j)=curTheta(j);
      }
    }
    /* stopping criteria */
    if(i>2 && curRotationQuality-rotationQuality2<1.0e-3)
    {
      break;
    }
    rotationQuality2=rotationQuality1;
    rotationQuality1=curRotationQuality;
  }

  for(int i=0; i<nData_; i++)
  {
    eigVecsRotated_.row(i).cwiseAbs().maxCoeff(&clusters_(i));
  }

  /* obtained quality */
  rotationQuality_=curRotationQuality;
}

template<typename T> T
RotationEvaluation<T>::evaluateQuality(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs)
{
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigVecsSquared;
  Eigen::Matrix<T, Eigen::Dynamic, 1> maxVals;
  eigVecsSquared.noalias()=eigVecs.array().square().matrix();
  maxVals.noalias()=eigVecsSquared.rowwise().maxCoeff();

  for(int i=0; i<nData_; i++)
  {
    if(maxVals(i)!=0.0)
    {
      eigVecsSquared.row(i)/=maxVals(i);
    }
  }
  return 1.0-(eigVecsSquared.sum()/nData_-1.0)/nDims_;
}

template<typename T> T
RotationEvaluation<T>::evaluateQualityGradient(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
  const Eigen::Matrix<T, Eigen::Dynamic, 1> &theta,
  const int ind)
{
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mtxV;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mtxU1, mtxU2;
  gradientU(theta, ind, mtxV);
  buildUab(theta, 0, ind-1, mtxU1);
  buildUab(theta, ind+1, nAngles_-1, mtxU2);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mtxA;
  mtxA.noalias()=eigVecs*mtxU1*mtxV*mtxU2;
  rotateEigenVectors(eigVecs, theta);

  Eigen::Matrix<T, Eigen::Dynamic, 1> maxVals(nData_);
  Eigen::VectorXi maxIndCol(nData_);
  for(int i=0; i<nData_; i++)
  {
    eigVecsRotated_.row(i).cwiseAbs().maxCoeff(&maxIndCol(i));
    maxVals(i)=eigVecsRotated_(i, maxIndCol(i));
  }

  /* gradient computation */
  T dA=0.0;
  T tmp1, tmp2, tmp3;
  for(int j=0; j<nDims_; j++)
  {
    for(int i=0; i<nData_; i++)
    {
      if(maxVals(i)!=0.0)
      {
        tmp3=maxVals(i)*maxVals(i);
        tmp1=mtxA(i, j)*eigVecsRotated_(i, j)/tmp3;
        tmp2=mtxA(i, maxIndCol(i))*(eigVecsRotated_(i, j)*eigVecsRotated_(i, j))/(tmp3*maxVals(i));
        dA+=tmp1-tmp2;
      }
    }
  }
  dA*=2.0/(nData_*nDims_);
  return dA;
}

template<typename T> void
RotationEvaluation<T>::rotateEigenVectors(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
  const Eigen::Matrix<T, Eigen::Dynamic, 1> &theta)
{
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mtxUab;
  buildUab(theta, 0, nAngles_-1, mtxUab);
  eigVecsRotated_.noalias()=eigVecs*mtxUab;
}

template<typename T> void
RotationEvaluation<T>::buildUab(
  const Eigen::Matrix<T, Eigen::Dynamic, 1> &theta,
  const int a,
  const int b,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mtxUab)
{
  mtxUab.resize(nDims_, nDims_);
  mtxUab.setIdentity();

  T cosTheta, sinTheta, uik;
  for(int k=a; k<=b; k++)
  {
    cosTheta=cosf(theta(k));
    sinTheta=sinf(theta(k));
    for(int i=0; i<nDims_; i++)
    {
      uik=mtxUab(i, ik_(k))*cosTheta-mtxUab(i, jk_(k))*sinTheta;
      mtxUab(i, jk_(k))=mtxUab(i, ik_(k))*sinTheta+mtxUab(i, jk_(k))*cosTheta;
      mtxUab(i, ik_(k))=uik;
    }
  }
}

template<typename T> void
RotationEvaluation<T>::gradientU(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &theta,
  const int ind,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mtxU)
{
  mtxU.resize(nDims_, nDims_);
  mtxU.setZero();
  T cosTheta=cosf(theta(ind));
  T sinTheta=sinf(theta(ind));
  mtxU(ik_(ind), ik_(ind))=-sinTheta;
  mtxU(ik_(ind), jk_(ind))=cosTheta;
  mtxU(jk_(ind), ik_(ind))=-cosTheta;
  mtxU(jk_(ind), jk_(ind))=-sinTheta;
}

template class RotationEvaluation<double>;
template class RotationEvaluation<float>;

}

}

