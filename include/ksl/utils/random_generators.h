#ifndef RANDOM_GENERATORS_H
#define RANDOM_GENERATORS_H

#include <cmath>
#include <cstdlib>
#include <Eigen/Core>

namespace ksl
{

namespace utils
{

template<typename T>
inline T
rnd(
  const T& minNum,
  const T& maxNum)
{
  T n;
  do
  {
    n=(static_cast<double>(std::rand())/RAND_MAX)*(maxNum-minNum)+minNum;
  } while(n>=maxNum || n<=minNum);
  return n;
}

template<typename T>
inline void
rnd(
  const T& minNum,
  const T& maxNum,
  const int nDim,
  Eigen::Matrix<T, Eigen::Dynamic, 1>& rndVec)
{
  rndVec.resize(nDim);
  for(int i=0; i<nDim; ++i)
  {
    rndVec(i)=rnd<T>(minNum, maxNum);
  }
}

template<typename T>
inline void
rnd(
  const T& minNum,
  const T& maxNum,
  const int nRows, const int nCols,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& rndMtx)
{
  rndMtx.resize(nRows, nCols);
  for(int i=0; i<nRows; ++i)
  {
    for(int j=0; j<nCols; ++j)
    {
      rndMtx(i, j)=rnd<T>(minNum, maxNum);
    }
  }
}

template<typename T>
inline T
normrnd(
  const T& mean,
  const T& stdDev)
{
  T u, v, s;
  do
  {
    u=rnd<T>(-1.0, 1.0);
    v=rnd<T>(-1.0, 1.0);
    s=u*u+v*v;
  } while(s>=1 || s==0);
  return mean+stdDev*u*std::sqrt(-2.0*std::log(s)/s);
}

template<typename T>
inline void
normrnd(
  const T& mean,
  const T& stdDev,
  const int nDim,
  Eigen::Matrix<T, Eigen::Dynamic, 1> &normrndVec)
{
  normrndVec.resize(nDim);
  for(int i=0; i<nDim; ++i)
  {
    normrndVec(i)=normrnd<T>(mean, stdDev);
  }
}

template<typename T>
inline void
normrnd(
  const T& mean,
  const T& stdDev,
  const int nRows, const int nCols,
  Eigen::MatrixXf &normrndMtx)
{
  normrndMtx.resize(nRows, nCols);
  for(int i=0; i<nRows; ++i)
  {
    for(int j=0; j<nCols; ++j)
    {
      normrndMtx(i, j)=normrnd<T>(mean, stdDev);
    }
  }
}

}

}

#endif // RANDOM_GENERATORS_H
