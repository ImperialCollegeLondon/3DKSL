#include <ksl/clustering/sparse_subspace_clustering.hpp>

namespace ksl
{

namespace clustering
{

SparseSubspaceClustering::SparseSubspaceClustering(void)
{}

SparseSubspaceClustering::~SparseSubspaceClustering(void)
{}

float
SparseSubspaceClustering::err(
  const int ind) const
{
  assert(0<=ind && ind<3);
  return err_[ind];
}

int
SparseSubspaceClustering::nGroups(void) const
{
  return nGroups_;
}

const Eigen::MatrixXf&
SparseSubspaceClustering::dataProj(void) const
{
  return dataProj_;
}

const Eigen::MatrixXf&
SparseSubspaceClustering::coeff(void) const
{
  return coeff_;
}

const Eigen::MatrixXf&
SparseSubspaceClustering::coeffP(void) const
{
  return coeffP_;
}

const Eigen::MatrixXf&
SparseSubspaceClustering::symGraph(void) const
{
  return symGraph_;
}

const Eigen::VectorXi&
SparseSubspaceClustering::groups(void) const
{
  return groups_;
}

void
SparseSubspaceClustering::compute(
  const Eigen::MatrixXf &data,
  const float alpha,
  const int r,
  const int affine,
  const int outlier,
  const int k,
  const float rho,
  const int nGroups,
  const int maxGroups,
  const float eigThresh,
  const int nIter,
  const float thr1,
  const float thr2)
{
  nData_=data.cols();
  nDims_=data.rows();
  float alphaAux;
  projectData(data, r);
  if(!outlier)
  {
    (alpha==0.0) ? (alphaAux=800.0) : (alphaAux=alpha);
    admmLasso(affine, alphaAux, nIter, thr1, thr2);
  }
  else
  {
    (alpha==0.0) ? (alphaAux=20.0) : (alphaAux=alpha);
    admmOutlier(affine, alphaAux, nIter, thr1, thr2);
  }
  buildAdjacencyMatrix(k, rho);
  SpectralClustering<float> c;
  nGroups_=c.compute(symGraph_, nGroups, eigThresh, maxGroups, 1000, 1e-3);
  groups_=c.clusters();
}

float
SparseSubspaceClustering::bestMap(
  const Eigen::VectorXi &groundTruth)
{
  assert(groundTruth.size()==nData_);

  Eigen::VectorXi groundTruthLbl, groupsLbl;
  igl::unique(groundTruth, groundTruthLbl);
  igl::unique(groups_, groupsLbl);
  int ngroundTruthLbl=groundTruthLbl.size();
  int ngroupsLbl=groupsLbl.size();
  int nLbl;
  (ngroundTruthLbl>ngroupsLbl) ? (nLbl=ngroundTruthLbl) : (nLbl=ngroupsLbl);

  Eigen::MatrixXf mtxG(nLbl, nLbl);
  Eigen::VectorXi indVec;
  mtxG.setZero();
  for(int i=0; i<ngroundTruthLbl; i++)
  {
    for(int j=0; j<ngroupsLbl; j++)
    {
      igl::find((groundTruth.array()==groundTruthLbl(i)).select(groups_.array()==groupsLbl(j), NULL), indVec);
      mtxG(i, j)=indVec.size();
    }
  }
  mtxG=(mtxG.maxCoeff()-mtxG.array()).matrix();

  Eigen::VectorXi mtxA;
  float c;
  hungarian(mtxG, mtxA, c);
	Eigen::VectorXi mtxAaux=mtxA;
  for(int i=0; i<mtxA.size(); i++)
  {
    for(int j=0; j<mtxA.size(); j++)
    {
      if(mtxA(j)==i)
      {
        mtxAaux(i)=j;
        break;
      }
    }
  }
  Eigen::VectorXi groups(groups_);
	groups_.fill(-1.0);
  for(int i=0; i<ngroupsLbl; i++)
  {
    igl::find(groups.array()==groupsLbl(i), indVec);
		if(mtxAaux(i)<ngroundTruthLbl)
		{
    	for(int j=0; j<indVec.size(); j++)
    	{
     		groups_(indVec(j))=groundTruthLbl(mtxA(i));
    	}
		}
  }

  return (float) (groundTruth.array()!=groups_.array()).cast<int>().sum()/nData_;
}

void
SparseSubspaceClustering::projectData(
  const Eigen::MatrixXf &data,
  const int r)
{
  /* no projection */
  if(r==0)
  {
    dataProj_=data;
    return;
  }
  /* projection using PCA */
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(data, Eigen::ComputeThinU | Eigen::ComputeThinV);
  dataProj_=svd.matrixU().topLeftCorner(nDims_, r).transpose()*data;
}

void
SparseSubspaceClustering::buildAdjacencyMatrix(
  const int k,
  const float rho)
{
  thrC(rho);

  Eigen::MatrixXf mtxCAbs(coeffP_.cwiseAbs());
  Eigen::MatrixXf mtxS(nData_, nData_);
  Eigen::MatrixXi mtxInd(nData_, nData_);

  /* normalization of the columns of coefficient matrix */
  igl::sort(mtxCAbs, 1, false, mtxS, mtxInd);
  if(k==0)
  {
    for(int i=0; i<nData_; i++)
    {
      mtxCAbs.col(i)/=mtxCAbs(mtxInd(0, i), i)+FLT_EPSILON;
    }
  }
  else
  {
    for(int i=0; i<nData_; i++)
    {
      for(int j=0; j<k; j++)
      {
        mtxCAbs(mtxInd(j, i), i)/=mtxCAbs(mtxInd(0, i), i)+FLT_EPSILON;
      }
    }
  }
  /* similarity graph */
  symGraph_=mtxCAbs+mtxCAbs.transpose();
}

void
SparseSubspaceClustering::admmLasso(
  const int affine,
  const float alpha,
  const int nIter,
  const float thr1,
  const float thr2)
{
  int i;
  float mu1=alpha/computeLambda(dataProj_);
  float mu2=alpha;
  Eigen::MatrixXf mtxA;
  Eigen::MatrixXf mtxC1(Eigen::MatrixXf::Zero(nData_, nData_));
  Eigen::MatrixXf mtxC2, mtxC3;
  Eigen::MatrixXf mtxLambda(Eigen::MatrixXf::Zero(nData_, nData_));
  Eigen::MatrixXf mtxZ;
  Eigen::MatrixXf mtxY(dataProj_.transpose()*dataProj_);
  Eigen::MatrixXf mtxOnes_mu2(Eigen::MatrixXf::Constant(nData_, nData_, 1.0/mu2));
  Eigen::MatrixXf mtxTmp(nData_, nData_);

  err_[0]=10.0*thr1;
  err_[1]=10.0*thr2;
  if(!affine)
  {
    mtxA=(mu1*mtxY+mu2*Eigen::MatrixXf::Identity(nData_, nData_)).inverse();

    for(i=0; i<nIter && err_[0]>thr1; i++)
    {
      /* updating Z */
      mtxZ=mtxA*(mu1*mtxY+mu2*mtxC1-mtxLambda);
      mtxZ.diagonal().setZero();
      /* updating C */
      mtxC3=mtxZ+mtxLambda/mu2;
      mtxC2=mtxC3.cwiseAbs()-mtxOnes_mu2;
      mtxC2=mtxC2.cwiseMax(0.0).cwiseProduct(mtxC3.cwiseSign());
      mtxC2.diagonal().setZero();
      /* updating Lagrange multipliers */
      mtxTmp=mtxZ-mtxC2;
      mtxLambda+=mu2*mtxTmp;
      /* updating errors */
      err_[0]=mtxTmp.cwiseAbs().maxCoeff();
      err_[1]=errorLinSys(dataProj_, mtxZ);
      mtxC1=mtxC2;
    }
    /*for(int j=0; j<2; j++)
    {
      std::cout<<"err_["<<j<<"]: "<<err_[j]<<std::endl;
    }
    std::cout<<"iter: "<<i<<std::endl;*/
  }
  else
  {
    Eigen::RowVectorXf lambda3(Eigen::RowVectorXf::Zero(nData_));
    Eigen::RowVectorXf onesRowVec(Eigen::RowVectorXf::Constant(nData_, 1.0));
    Eigen::VectorXf onesVec(Eigen::VectorXf::Constant(nData_, 1.0));
    mtxA=(mu1*mtxY+mu2*(Eigen::MatrixXf::Identity(nData_, nData_)+Eigen::MatrixXf::Constant(nData_, nData_, 1.0))).inverse();

    err_[2]=10.0*thr1;
    for(i=0; i<nIter && (err_[0]>thr1 || err_[2]>thr1); i++)
    {
      /* updating Z */
      mtxZ=mtxA*(mu1*mtxY+mu2*mtxC1-mtxLambda+mu2*onesVec*(onesRowVec-lambda3/mu2));
      mtxZ.diagonal().setZero();
      /* updating C */
      mtxC3=mtxZ+mtxLambda/mu2;
      mtxC2=mtxC3.cwiseAbs()-mtxOnes_mu2;
      mtxC2=mtxC2.cwiseMax(0.0).cwiseProduct(mtxC3.cwiseSign());
      mtxC2.diagonal().setZero();
      /* updating Lagrange multipliers */
      mtxTmp=mtxZ-mtxC2;
      mtxLambda+=mu2*mtxTmp;
      lambda3+=mu2*(onesRowVec*mtxZ-onesRowVec);
      /* computing errors */
      err_[0]=mtxTmp.cwiseAbs().maxCoeff();
      err_[1]=errorLinSys(dataProj_, mtxZ);
      err_[2]=(onesRowVec*mtxZ-onesRowVec).cwiseAbs().maxCoeff();
      mtxC1=mtxC2;
    }
    /*for(int j=0; j<3; j++)
    {
      std::cout<<"err_["<<j<<"]: "<<err_[j]<<std::endl;
    }
    std::cout<<"iter: "<<i<<std::endl;*/
  }
  coeff_=mtxC1;
}

void
SparseSubspaceClustering::admmOutlier(
  const int affine,
  const float alpha,
  const int nIter,
  const float thr1,
  const float thr2)
{
  float gamma=alpha/dataProj_.cwiseAbs().colwise().sum().maxCoeff();
  Eigen::MatrixXf mtxP(nDims_, nData_+nDims_);
  mtxP<<dataProj_, Eigen::MatrixXf::Identity(nDims_, nDims_)/gamma;
  
  int i;
  float mu1=alpha/computeLambda(mtxP);
  float mu2=alpha;
  Eigen::MatrixXf mtxA;
  Eigen::MatrixXf mtxC1(Eigen::MatrixXf::Zero(nData_+nDims_, nData_));
  Eigen::MatrixXf mtxC2, mtxC3;
  Eigen::MatrixXf mtxLambda1(Eigen::MatrixXf::Zero(nDims_, nData_));
  Eigen::MatrixXf mtxLambda2(Eigen::MatrixXf::Zero(nData_+nDims_, nData_));
  Eigen::MatrixXf mtxZ;
  Eigen::MatrixXf mtxOnes_mu2(Eigen::MatrixXf::Constant(nData_+nDims_, nData_, 1.0/mu2));
  Eigen::MatrixXf mtxTmp(nData_+nDims_, nData_);

  err_[0]=10.0*thr1;
  err_[1]=10.0*thr2;
  if(!affine)
  {
    mtxA=(mu1*mtxP.transpose()*mtxP+mu2*Eigen::MatrixXf::Identity(nData_+nDims_, nData_+nDims_)).inverse();

    for(i=0; i<nIter && (err_[0]>thr1 || err_[1]>thr2); i++)
    {
      /* updating Z */
      mtxZ=mtxA*(mu1*mtxP.transpose()*(dataProj_+mtxLambda1/mu1)+mu2*mtxC1-mtxLambda2);
      mtxZ.topLeftCorner(nData_, mtxZ.cols()).diagonal().setZero();
      /* updating C */
      mtxC3=mtxZ+mtxLambda2/mu2;
      mtxC2=mtxC3.cwiseAbs()-mtxOnes_mu2;
      mtxC2=mtxC2.cwiseMax(0.0).cwiseProduct(mtxC3.cwiseSign());
      mtxC2.topLeftCorner(nData_, mtxC2.cols()).diagonal().setZero();
      /* updating Lagrange multipliers */
      mtxTmp=mtxZ-mtxC2;
      mtxLambda1+=mu1*(dataProj_-mtxP*mtxZ);
      mtxLambda2+=mu2*mtxTmp;
      /* updating errors */
      err_[0]=mtxTmp.cwiseAbs().maxCoeff();
      err_[1]=errorLinSys(mtxP, mtxZ);
      mtxC1=mtxC2;
    }
    /*for(int j=0; j<2; j++)
    {
      std::cout<<"err_["<<j<<"]: "<<err_[j]<<std::endl;
    }
    std::cout<<"iter: "<<i<<std::endl;*/
  }
  else
  {
    Eigen::VectorXf delta(nData_+nDims_);
    delta<<Eigen::VectorXf::Constant(nData_, 1.0),
          Eigen::VectorXf::Zero(nDims_);
    Eigen::RowVectorXf lambda3(Eigen::RowVectorXf::Zero(nData_));
    Eigen::RowVectorXf onesRowVec(Eigen::RowVectorXf::Constant(nData_, 1.0));
    mtxA=(mu1*mtxP.transpose()*mtxP+mu2*(Eigen::MatrixXf::Identity(nData_+nDims_, nData_+nDims_)+delta*delta.transpose())).inverse();

    err_[2]=10.0*thr1;
    for(i=0; i<nIter && (err_[0]>thr1 || err_[1]>thr2 || err_[2]>thr1); i++)
    {
      /* updating Z */
      mtxZ=mtxA*(mu1*mtxP.transpose()*(dataProj_+mtxLambda1/mu1)+mu2*mtxC1-mtxLambda2+mu2*delta*(onesRowVec-lambda3/mu2));
      mtxZ.topLeftCorner(nData_, mtxZ.cols()).diagonal().setZero();
      /* updating C */
      mtxC3=mtxZ+mtxLambda2/mu2;
      mtxC2=mtxC3.cwiseAbs()-mtxOnes_mu2;
      mtxC2=mtxC2.cwiseMax(0.0).cwiseProduct(mtxC3.cwiseSign());
      mtxC2.topLeftCorner(nData_, mtxC2.cols()).diagonal().setZero();
      /* updating Lagrange multipliers */
      mtxTmp=mtxZ-mtxC2;
      mtxLambda1+=mu1*(dataProj_-mtxP*mtxZ);
      mtxLambda2+=mu2*mtxTmp;
      lambda3+=mu2*(delta.transpose()*mtxZ-onesRowVec);
      /* updating errors */
      err_[0]=mtxTmp.cwiseAbs().maxCoeff();
      err_[1]=errorLinSys(mtxP, mtxZ);
      err_[2]=(delta.transpose()*mtxZ-onesRowVec).cwiseAbs().maxCoeff();
      mtxC1=mtxC2;
    }
    /*for(int j=0; j<3; j++)
    {
      std::cout<<"err_["<<j<<"]: "<<err_[j]<<std::endl;
    }
    std::cout<<"iter: "<<i<<std::endl;*/
  }
  coeff_=mtxC1.topRows(nData_);
}

float
SparseSubspaceClustering::computeLambda(
  const Eigen::MatrixXf &mtx)
{
  Eigen::MatrixXf mtxT1((mtx.transpose()*dataProj_).topRows(nData_));
  Eigen::MatrixXf mtxT2(mtxT1.diagonal().asDiagonal());
  mtxT1=(mtxT1-mtxT2).cwiseAbs();
  
  return mtxT1.colwise().maxCoeff().minCoeff();
}

float
SparseSubspaceClustering::errorLinSys(
  const Eigen::MatrixXf &mtx1,
  const Eigen::MatrixXf &mtx2)
{
  int rows1=mtx1.rows();
  int cols1=mtx1.cols();
  int rows2=mtx2.rows();
  int cols2=mtx2.cols();
  Eigen::MatrixXf mtxY, mtxY0, mtxC;

  if(rows2>cols2)
  {
    Eigen::MatrixXf mtxE(mtx1.block(0, cols2, rows1, cols1-cols2)*mtx2.block(cols2, 0, rows2-cols2, cols2));
    mtxY=mtx1.topLeftCorner(rows1, cols2);
    mtxY0=mtxY-mtxE;
    mtxC=mtx2.topLeftCorner(cols2, cols2);
  }
  else
  {
    mtxY=mtx1;
    mtxY0=mtx1;
    mtxC=mtx2;
  }

  /* Y0 matrix normalization */
  Eigen::MatrixXf mtxYn(mtxY0.rows(), cols2);
  Eigen::RowVectorXf n(cols2);
  for(int i=0; i<cols2; i++)
  {
    n(i)=mtxY0.col(i).norm();
    mtxYn.col(i)=mtxY0.col(i)/n(i);
  }

  Eigen::MatrixXf mtxM(n.replicate(mtxY.rows(), 1));
  Eigen::MatrixXf mtxS(mtxYn-(mtxY*mtxC).cwiseQuotient(mtxM));
  return sqrt(mtxS.array().square().colwise().sum().maxCoeff());
}

void
SparseSubspaceClustering::thrC(
  const float rho)
{
  if(rho==1.0)
  {
    coeffP_=coeff_;
    return;
  }

  Eigen::MatrixXf mtxS(nData_, nData_);
  Eigen::MatrixXi mtxInd(nData_, nData_);
  float c, cSum;
  coeffP_=Eigen::MatrixXf::Zero(nData_, nData_);
  igl::sort(coeff_.cwiseAbs(), 1, false, mtxS, mtxInd);
  for(int i=0; i<nData_; i++)
  {
    c=mtxS.col(i).sum();
    cSum=0.0;
    for(int j=0; cSum<rho*c; j++)
    {
      cSum+=mtxS(j, i);
      coeffP_(mtxInd(j, i), i)=coeff_(mtxInd(j, i), i);
    }
  }
}

}

}

