#ifndef ROTATION_EVALUATION_HPP
#define ROTATION_EVALUATION_HPP

#include <Eigen/Core>
#include <math.h>

namespace ksl
{

namespace clustering
{

template<typename T>
class RotationEvaluation
{

private:

  int nData_;
  int nDims_;
  int nAngles_;
  Eigen::VectorXi ik_;
  Eigen::VectorXi jk_;

protected:

  T rotationQuality_;
  Eigen::VectorXi clusters_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigVecsRotated_;

public:

  RotationEvaluation(void);
  virtual ~RotationEvaluation(void);

  T
  rotationQuality(void) const
  {
    return rotationQuality_;
  }
  const Eigen::VectorXi&
  clusters(void) const
  {
    return clusters_;
  }
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&
  eigVecsRotated(void) const
  {
    return eigVecsRotated_;
  }

  virtual void
  compute(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
    const int nIter=200);

protected:

  virtual void
  evaluateRotation(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
    const int nIter=200);
  virtual T
  evaluateQuality(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs);
  virtual T
  evaluateQualityGradient(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &theta,
    const int ind);

  virtual void
  rotateEigenVectors(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigVecs,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &theta);

  virtual void
  buildUab(
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &theta,
    const int a,
    const int b,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mtxUab);
  virtual void
  gradientU(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &theta,
    const int ind,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mtxU);

private:

};

}

}

#endif // ROTATION_EVALUATION_HPP

