#include <ksl/clustering/random_generators.hpp>

namespace ksl
{

namespace clustering
{

float
rnd(
  const float minNum,
  const float maxNum)
{
  float n;
  do
  {
    n=((float) rand()/(float) RAND_MAX)*(maxNum-minNum)+minNum;
  } while(n>=maxNum || n<=minNum);
  return n;
}

void
rnd(
  const float minNum,
  const float maxNum,
  const int dim,
  Eigen::VectorXf &rndVec)
{
  rndVec.resize(dim);
  for(int i=0; i<dim; i++)
  {
    rndVec(i)=rnd(minNum, maxNum);
  }
}

void
rnd(
  const float minNum,
  const float maxNum,
  const int rows,
  const int cols,
  Eigen::MatrixXf &rndMtx)
{
  rndMtx.resize(rows, cols);
  for(int i=0; i<rows; i++)
  {
    for(int j=0; j<cols; j++)
    {
      rndMtx(i, j)=rnd(minNum, maxNum);
    }
  }
}

float
normrnd(
  const float mean,
  const float stdDev)
{
  float u, v, s;
  do
  {
    u=rnd(-1.0, 1.0);
    v=rnd(-1.0, 1.0);
    s=u*u+v*v;
  } while(s>=1 || s==0);
  return mean+stdDev*u*sqrtf(-2.0*logf(s)/s);
}

void
normrnd(
  const float mean,
  const float stdDev,
  const int dim,
  Eigen::VectorXf &normrndVec)
{
  normrndVec.resize(dim);
  for(int i=0; i<dim; i++)
  {
    normrndVec(i)=normrnd(mean, stdDev);
  }
}

void
normrnd(
  const float mean,
  const float stdDev,
  const int rows,
  const int cols,
  Eigen::MatrixXf &normrndMtx)
{
  normrndMtx.resize(rows, cols);
  for(int i=0; i<rows; i++)
  {
    for(int j=0; j<cols; j++)
    {
      normrndMtx(i, j)=normrnd(mean, stdDev);
    }
  }
}

}

}

