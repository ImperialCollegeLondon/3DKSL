#ifndef DATASET_H
#define DATASET_H

#include <algorithm>
#include <cassert>
#include <Eigen/Core>

namespace ksl
{

namespace utils
{

template<typename T>
inline void
cov(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mtx,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& covMtx)
{
  if(mtx.rows()>1)
  {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cMtx(mtx.rowwise()-mtx.colwise().mean());
    covMtx.noalias()=(cMtx.transpose()*cMtx)/(mtx.rows()-1);
  }
  else
  {
    covMtx=mtx;
  }
}

template<typename T>
inline void
createSyntheticDataset(
  const int nPoints,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dataset,
  const T& a=1.0, const T& b=4.0, const T& c=21.0)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> tVec(
    ((b*b-a*a)*0.5*(Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(nPoints)+
    Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(nPoints)).array()+a*a).cwiseSqrt());
  tVec*=M_PI;
  dataset.resize(3, nPoints);
  dataset.row(0)=tVec.array()*tVec.array().cos();
  dataset.row(1)=c*Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(nPoints);
  dataset.row(2)=tVec.array()*tVec.array().sin();
}

template<typename T>
inline T
median(
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec)
{
  assert(vec.size()>0);

  Eigen::Matrix<T, Eigen::Dynamic, 1> aVec(vec);
  const int vecSize=aVec.size(), vecHalfSize=0.5*vecSize;
  std::nth_element(aVec.data(), aVec.data()+vecHalfSize, aVec.data()+vecSize);
  if(vecSize%2==0)
  {
    std::nth_element(aVec.data(), aVec.data()+vecHalfSize-1, aVec.data()+vecSize);
    return 0.5*(aVec(vecHalfSize)+aVec(vecHalfSize-1));
  }
  return aVec(vecHalfSize);
}

template<typename T>
inline void
pairwiseDistanceEuclidean(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mtx1,
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mtx2,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& pairwiseDist)
{
  assert(mtx1.cols()==mtx2.cols());

  pairwiseDist.noalias()=mtx1.array().square().rowwise().sum().matrix()*Eigen::Matrix<T, 1, Eigen::Dynamic>::Ones(mtx2.rows());
  pairwiseDist.noalias()+=Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(mtx1.rows())*mtx2.array().square().rowwise().sum().matrix().transpose();
  pairwiseDist.noalias()-=2.0*mtx1*mtx2.transpose();
  pairwiseDist=pairwiseDist.cwiseSqrt();
  pairwiseDist.diagonal().setZero();
}

template<typename T>
inline void
pairwiseDistanceEuclidean(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mtx,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& pairwiseDist)
{
  pairwiseDistanceEuclidean<T>(mtx, mtx, pairwiseDist);
}

template<typename T>
inline void
rescaleCentre(
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mtx)
{
  mtx.noalias()-=(mtx.rowwise().sum()/mtx.cols())*Eigen::Matrix<T, 1, Eigen::Dynamic>::Ones(mtx.cols());
  mtx/=(mtx.cwiseAbs()+1.0e-10).maxCoeff();
}

}

}

#endif // DATASET_H
