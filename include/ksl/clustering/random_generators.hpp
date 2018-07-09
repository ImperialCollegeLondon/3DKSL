#ifndef RANDOM_GENERATORS_HPP
#define RANDOM_GENERATORS_HPP

#include <Eigen/Core>
#include <math.h>
#include <stdlib.h>

namespace ksl
{

namespace clustering
{

float
rnd(
  const float minNum,
  const float maxNum);
void
rnd(
  const float minNum,
  const float maxNum,
  const int dim,
  Eigen::VectorXf &rndVec);
void
rnd(
  const float minNum,
  const float maxNum,
  const int rows,
  const int cols,
  Eigen::MatrixXf &rndMtx);

float
normrnd(
  const float mean,
  const float stdDev);
void
normrnd(
  const float mean,
  const float stdDev,
  const int dim,
  Eigen::VectorXf &normrndVec);
void
normrnd(
  const float mean,
  const float stdDev,
  const int rows,
  const int cols,
  Eigen::MatrixXf &normrndMtx);

}

}

#endif // RANDOM_GENERATORS_HPP

