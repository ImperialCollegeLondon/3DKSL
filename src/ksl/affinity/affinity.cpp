#include <ksl/affinity/affinity.h>

namespace ksl
{

namespace affinity
{

template<typename T>
Affinity<T>::Affinity(void)
{}

template<typename T>
Affinity<T>::Affinity(
  const int& nPoints, const int& nDims):
  bparams_(nPoints, nDims)
{}

template<typename T>
Affinity<T>::Affinity(
  const AffinityParams<T>& params):
  bparams_(params)
{}

template<typename T>
Affinity<T>::~Affinity(void)
{}

template<typename T>
void
Affinity<T>::compute(
  const mtxT& dMtx)
{
  bparams_.nPoints=dMtx.rows(), bparams_.nDims=dMtx.cols();
  this->computeAffinity(dMtx);
}

template<typename T>
void
Affinity<T>::compute(
  const mtxT& dMtx,
  const vecI& indVec)
{
  assert(dMtx.rows()==indVec.size());
  this->computeAffinity(dMtx, indVec);
}

template class Affinity<double>;
template class Affinity<float>;

}

}
