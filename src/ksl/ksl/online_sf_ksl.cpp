#include <ksl/ksl/online_sf_ksl.h>
#include <iostream>

namespace ksl
{

OnlineKinematicStructure::OnlineKinematicStructure(
  const int& width, const int& height,
  const float& scale,
  const utils::CamParams<float>& camParams,
  const float& beta,
  const affinity::NonMetricTwoViewsParams<float>& aParams,
  const clustering::LabelPropagationParams<float>& cParams):
  //tTrackingPoints_(0.0), tDist_(0.0), tAffinity_(0.0), tCluster_(0.0), tKS_(0.0),
  indToogle_(true),
  scale_(scale),
  camParams_(camParams),
  rows_(camParams_.height/scale_),
  frame_(0),
  nPoints_(0), nLostPoints_(0),
  resolution_(0.0), beta_(beta), //beta_(0.275),
  pcloudOriginal_(new pcl::PointCloud<pcl::PointXYZ>(camParams_.width, camParams_.height)),
  pcloud_(new pcl::PointCloud<pcl::PointXYZ>),
  pcloudSubSampled_(new pcl::PointCloud<pcl::PointXYZ>),
  pointsMap_(nullptr, 0, 0),
  pointsPairwiseDistanceEuclidean_(2),
  sceneFlow_(camParams_.width, camParams_.height, rows_),
  a_(aParams),
  c_(cParams),
  yMtx_(c_.yMtx())
{
  assert(camParams_.width>0 && camParams_.height>0);
  initKSGraph();
}

OnlineKinematicStructure::OnlineKinematicStructure(
  const int& width, const int& height,
  const float& scale,
  const utils::CamParams<float>& camParams,
  const float& beta,
  const affinity::NonMetricTwoViewsParams<float>& aParams,
  const clustering::LabelPropagationParams<float>& cParams,
  const int& nPoints):
  //tTrackingPoints_(0.0), tDist_(0.0), tAffinity_(0.0), tCluster_(0.0), tKS_(0.0),
  indToogle_(true),
  scale_(scale),
  camParams_(camParams),
  rows_(camParams_.height/scale_),
  frame_(0),
  nPoints_(nPoints), nLostPoints_(0),
  resolution_(0.0), beta_(beta),
  pcloudOriginal_(new pcl::PointCloud<pcl::PointXYZ>(camParams_.width, camParams_.height)),
  pcloud_(new pcl::PointCloud<pcl::PointXYZ>),
  pcloudSubSampled_(new pcl::PointCloud<pcl::PointXYZ>),
  pointsMap_(nullptr, 0, 0),
  pointsPairwiseDistanceEuclidean_(2),
  sceneFlow_(camParams_.width, camParams_.height, rows_),
  a_(aParams),
  c_(cParams),
  yMtx_(c_.yMtx())
{
  assert(camParams_.width>0 && camParams_.height>0 && nPoints_>0);
  initKSGraph();
}

OnlineKinematicStructure::OnlineKinematicStructure(
  const int& width, const int& height,
  const float& scale,
  const utils::CamParams<float>& camParams,
  const float& beta,
  const affinity::NonMetricTwoViewsParams<float>& aParams,
  const clustering::LabelPropagationParams<float>& cParams,
  const float& resolution):
  //tTrackingPoints_(0.0), tDist_(0.0), tAffinity_(0.0), tCluster_(0.0), tKS_(0.0),
  indToogle_(true),
  scale_(scale),
  camParams_(camParams),
  rows_(camParams_.height/scale_),
  frame_(0),
  nPoints_(0), nLostPoints_(0),
  resolution_(resolution), beta_(beta),
  pcloudOriginal_(new pcl::PointCloud<pcl::PointXYZ>(camParams_.width, camParams_.height)),
  pcloud_(new pcl::PointCloud<pcl::PointXYZ>),
  pcloudSubSampled_(new pcl::PointCloud<pcl::PointXYZ>),
  pointsMap_(nullptr, 0, 0),
  pointsPairwiseDistanceEuclidean_(2),
  sceneFlow_(camParams_.width, camParams_.height, rows_),
  a_(aParams),
  c_(cParams),
  yMtx_(c_.yMtx())
{
  assert(camParams_.width>0 && camParams_.height>0 && resolution_>0.0);
  initKSGraph();
}

OnlineKinematicStructure::~OnlineKinematicStructure(void)
{
  destroyKSGraph();
}

void
OnlineKinematicStructure::computeKSL(
  const int& opt,
  const int& minEvidence)
{
  computeDist(opt, minEvidence);

  const int nPoints=nPoints_-nLostPoints_;
  if(nPoints<nMinPoints_)
  {
    return;
  }

  Eigen::VectorXi indVec(nPoints);
  Eigen::MatrixXf dDist(dDist_);
  yMtx_.resize(nPoints, ylMtx_.cols());

  int kk=0;
  for(int k=0; k<nPoints_; ++k)
  {
    if(accTrackPoints_(k)>minEvidence)
    {
      if(kk!=k)
      {
        dDist.row(kk).swap(dDist.row(k));
        dDist.col(kk).swap(dDist.col(k));
      }
      yMtx_.row(kk)=ylMtx_.row(k);
      indVec(kk)=k;
      ++kk;
    }
  }

  Eigen::Map<Eigen::MatrixXf, 0, Eigen::OuterStride<> > dMtx(dDist.data(),
    nPoints, nPoints, Eigen::OuterStride<>(nPoints_));

  c_.compute(dMtx, &a_);

  //tAffinity_=a_.tAffinity();
  //tCluster_=c_.tCluster();

  if(a_.rdMtx().rows()==nPoints && a_.rdMtx().cols()==nPoints)
  {
    dMtx=a_.rdMtx();
  }

  //t_.tic();

  Eigen::MatrixXf gCount(c_.nClusters(), c_.nClusters());
  Eigen::VectorXf cCount(c_.nClusters());
  centroids_.setZero(c_.nClusters(), 3), gCount.setZero(), cCount.setZero();
  gMtx_.setZero(c_.nClusters(), c_.nClusters());
  for(int k=0; k<nPoints; ++k)
  {
    const int kInd=c_.clusters()(k);
    for(int l=k+1; l<nPoints; ++l)
    {
      const int lInd=c_.clusters()(l);
      gMtx_(kInd, lInd)+=dMtx(k, l), gMtx_(lInd, kInd)+=dMtx(l, k);
      ++gCount(kInd, lInd), ++gCount(lInd, kInd);
    }
    centroids_.row(kInd)+=pointsMap_.col(indVec(k)).head<3>();
    ++cCount(kInd);
  }
  gMtx_=gMtx_.cwiseQuotient(gCount);
  gMtx_*=beta_/gMtx_.maxCoeff();

  for(int k=0; k<c_.nClusters(); ++k)
  {
    centroids_.row(k)/=cCount(k);
  }
  Eigen::MatrixXf centroidsPairwiseDistanceEuclidean;
  utils::pairwiseDistanceEuclidean<float>(centroids_, centroidsPairwiseDistanceEuclidean);

  Eigen::MatrixXf lapMtx, bhDist(c_.nClusters(), c_.nClusters());
  clustering::laplacian<float>(((-centroidsPairwiseDistanceEuclidean.array().square())).exp(), lapMtx);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(lapMtx);
  const Eigen::MatrixXf& eigvectors(eig.eigenvectors());
  const Eigen::VectorXf& eigvals(eig.eigenvalues());
  bhDist.setZero();
  for(int i=0; i<c_.nClusters(); ++i)
  {
    for(int j=i+1; j<c_.nClusters(); ++j)
    {
      for(int k=0; k<c_.nClusters()-1; ++k)
      {
        bhDist(i, j)+=std::pow((eigvectors(i, k)-eigvectors(j, k))/(1.0-eigvals(k)), 2.0);
      }
      bhDist(j, i)=bhDist(i, j);
    }
  }
  bhDist=bhDist.cwiseSqrt();
  bhDist*=(1.0-beta_)/bhDist.maxCoeff();
  gMtx_+=bhDist;

  //tKS_=t_.toc();

  if(a_.rdMtx().rows()==nPoints && a_.rdMtx().cols()==nPoints)
  {
    kk=nPoints-1;
    for(int k=nPoints_-1; k>=0; --k)
    {
      if(accTrackPoints_(k)>minEvidence)
      {
        if(kk!=k)
        {
          dDist.row(kk).swap(dDist.row(k));
          dDist.col(kk).swap(dDist.col(k));
        }
        --kk;
      }
    }
    dDist_=dDist;
  }

  const int diffClusters=c_.nClusters()-ylMtx_.cols();
  if(diffClusters!=0)
  {
    ylMtx_.conservativeResize(Eigen::NoChange, c_.nClusters());
    if(diffClusters>0)
    {
      ylMtx_.rightCols(diffClusters).setZero();
    }
  }
  kk=0;
  for(int k=0; k<nPoints_; ++k)
  {
    if(accTrackPoints_(k)>minEvidence)
    {
      ylMtx_.row(k)=yMtx_.row(kk);
      ++kk;
    }
  }

  createKSGraph();
}

void
OnlineKinematicStructure::computeSceneFlow(
  const cv::Mat& img1, const cv::Mat& img2,
  const cv::Mat& depth1, const cv::Mat& depth2)
{
  assert(img1.cols==camParams_.width && depth1.cols==camParams_.width);
  assert(img1.rows==camParams_.height && depth1.rows==camParams_.height);
  assert(img2.cols==camParams_.width && depth2.cols==camParams_.width);
  assert(img2.rows==camParams_.height && depth2.rows==camParams_.height);

  cv::Mat gray1, gray2;
  cv::cvtColor(img1, gray1, cv::COLOR_RGB2GRAY);
  cv::cvtColor(img2, gray2, cv::COLOR_RGB2GRAY);
  sceneFlow_.compute(gray1, gray2, depth1, depth2);

   /* first frame */
  if(frame_==0)
  {
    utils::depthToPCloud<float>(img1, depth1, camParams_, pcloudOriginal_);
    utils::fastBilateralFilter<float, pcl::PointXYZ>(10.0, 10.0e-3, pcloudOriginal_);
    subSamplePCloud();
    computePointsPairwiseDistanceEuclidean();
  }
  ++frame_;
}

void
OnlineKinematicStructure::trackPoints(
  const cv::Mat& img,
  const cv::Mat& depth,
  const int& minEvidence,
  const float& dEps)
{
  utils::depthToPCloud<float>(img, depth, camParams_, pcloudOriginal_);
  //utils::fastBilateralFilter<float, pcl::PointXYZ>(10.0, 10.0e-3, pcloudOriginal_);

  //t_.tic();

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcloudSubSampled(new pcl::PointCloud<pcl::PointXYZ>);
  subSamplePCloud(pcloudSubSampled);
  /* octrees representations to deal with lost points and keep an evenly density */
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution_);
  octree.setInputCloud(pcloud_);
  octree.addPointsFromInputCloud();
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octreeSubSampled(resolution_);
  octreeSubSampled.setInputCloud(pcloudSubSampled_);
  octreeSubSampled.addPointsFromInputCloud();
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octreeSampled(resolution_);
  octreeSampled.setInputCloud(pcloudSubSampled);
  octreeSampled.addPointsFromInputCloud();

  const int nPoints=pcloudSubSampled->size();
  int meanNeighbours=0;
  Eigen::VectorXi neighbours(nPoints);
  std::vector<int> indVec;
  std::vector<float> distVec;
  for(int k=0; k<nPoints; ++k)
  {
    neighbours(k)=octreeSubSampled.radiusSearch(
      pcloudSubSampled->at(k), resolution_, indVec, distVec);
    meanNeighbours+=neighbours(k);
  }
  meanNeighbours=std::ceil(2.0*meanNeighbours/nPoints);

  /* mapping interface */
  const Eigen::Map<const Eigen::VectorXf> dxp(sceneFlow_.dxp(), sceneFlow_.size());
  const Eigen::Map<const Eigen::VectorXf> dyp(sceneFlow_.dyp(), sceneFlow_.size());
  const Eigen::Map<const Eigen::VectorXf> dzp(sceneFlow_.dzp(), sceneFlow_.size());

  //sceneFlow_.createImg();
  //sceneFlow_.showImg();

  indVec.resize(1), distVec.resize(1);
  nLostPoints_=nPoints_;
  const float invScale=1.0/scale_;
  const int nNeigh=scale_-1.0;
  for(int k=0; k<nPoints_; ++k)
  {
    int i, j, lostPoint=1;
    int indVecMin=nPoints;
    utils::xyzToIJ<float>(
      pcloudSubSampled_->at(k).x, pcloudSubSampled_->at(k).y, pcloudSubSampled_->at(k).z,
      camParams_, &i, &j);
    indVec[0]=nPoints;
    if(0<=i && i<camParams_.height && 0<=j && j<camParams_.width)
    {
      int ind=i*invScale+std::round(j*invScale)*rows_;
      float dx=dxp(ind), dy=dyp(ind), dz=dzp(ind);

      int n=1;
      for(int ii=i-nNeigh; ii<=i+nNeigh; ++ii)
      {
        for(int jj=j-nNeigh; jj<=j+nNeigh; ++jj)
        {
          ind=ii*invScale+std::round(jj*invScale)*rows_;
          if(0<=ind && ind<sceneFlow_.size())
          {
            dx+=dxp(ind), dy+=dyp(ind), dz+=dzp(ind);
            ++n;
          }
        }
      }
      const float scaledN=1.0/n;
      pcloudSubSampled_->at(k).x+=dy*scaledN;
      pcloudSubSampled_->at(k).y-=dz*scaledN;
      pcloudSubSampled_->at(k).z+=dx*scaledN;

      octree.nearestKSearch(pcloudSubSampled_->at(k), 1, indVec, distVec);
      indVecMin=indVec[0];
      if(std::abs(pcloud_->at(indVecMin).x-pcloudSubSampled_->at(k).x)<dEps &&
        std::abs(pcloud_->at(indVecMin).y-pcloudSubSampled_->at(k).y)<dEps &&
        std::abs(pcloud_->at(indVecMin).z-pcloudSubSampled_->at(k).z)<dEps)
      {
        octreeSampled.nearestKSearch(pcloudSubSampled_->at(k), 1, indVec, distVec);
        if(neighbours(indVec[0])<=meanNeighbours)
        {
          pcloudSubSampled_->at(k)=pcloud_->at(indVecMin);
          lostPoint=0;
          ++accTrackPoints_(k);
          if(accTrackPoints_(k)>minEvidence)
          {
            --nLostPoints_;
          }
        }
      }
    }

    /* lost points */
    if(lostPoint==1)
    {
      int indMin;
      accTrackPoints_(k)=0;
      neighbours.minCoeff(&indMin);
      ++neighbours(indMin);
      if(indVec[0]<nPoints && indVecMin!=indVec[0])
      {
        --neighbours(indVec[0]);
      }
      pcloudSubSampled_->at(k)=pcloudSubSampled->at(indMin);
    }
  }

  //tTrackingPoints_=t_.toc();
}

void
OnlineKinematicStructure::computePointsPairwiseDistanceEuclidean(void)
{
  mapPoints(pcloudSubSampled_);
  indToogle_^=true;
  utils::pairwiseDistanceEuclidean<float>(pointsMap_.topRows<3>().transpose(),
    pointsPairwiseDistanceEuclidean_[indToogle_]);
  pointsPairwiseDistanceEuclidean_[indToogle_]*=(1.0/pointsPairwiseDistanceEuclidean_[indToogle_].norm());
}

void
OnlineKinematicStructure::initKSGraph(
  const int& nVertices)
{
  igraph_empty(&ksGraph_, nVertices, IGRAPH_UNDIRECTED);
}

void
OnlineKinematicStructure::createKSGraph(void)
{
  //t_.tic();

  destroyKSGraph();
  initKSGraph(c_.nClusters());

  const int nEdges=c_.nClusters()*(c_.nClusters()-1);
  igraph_vector_t edges, ks, weights;
  igraph_vector_init(&edges, nEdges);
  igraph_vector_init(&ks, 0);
  igraph_vector_init(&weights, nEdges>>1);
  for(int k=0, i=0, j=i+1; k<nEdges; k+=2)
  {
    VECTOR(edges)[k]=i;
    VECTOR(edges)[k+1]=j;
    VECTOR(weights)[k>>1]=gMtx_(i, j);
    ++j;
    if(j>=c_.nClusters())
    {
      ++i;
      j=i+1;
    }
  }
  igraph_add_edges(&ksGraph_, &edges, nullptr);

  igraph_minimum_spanning_tree(&ksGraph_, &ks, &weights);
  ks_.resize(igraph_vector_size(&ks), 2);
  for(int k=0; k<igraph_vector_size(&ks); ++k)
  {
    const int kInd=2*VECTOR(ks)[k];
    ks_(k, 0)=VECTOR(edges)[kInd], ks_(k, 1)=VECTOR(edges)[kInd+1];;
  }

  igraph_vector_destroy(&edges);
  igraph_vector_destroy(&ks);
  igraph_vector_destroy(&weights);

  //tKS_+=t_.toc();
}

void
OnlineKinematicStructure::destroyKSGraph(void)
{
  igraph_destroy(&ksGraph_);
}

void
OnlineKinematicStructure::mapPoints(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pcloud)
{
  new (&pointsMap_) Eigen::Map<const Eigen::MatrixXf>(
    pcloud->getMatrixXfMap(3, 4, 0).data(), 4, pcloud->size());
}

void
OnlineKinematicStructure::subSamplePCloud(void)
{
  if(nPoints_>0)
  {
    utils::subSamplePCloud<float, pcl::PointXYZ>(pcloudOriginal_, nPoints_,
      pcloud_, pcloudSubSampled_);
    resolution_=0.02;
  }
  else if(resolution_>0.0)
  {
    utils::subSamplePCloud<float, pcl::PointXYZ>(pcloudOriginal_, resolution_,
      pcloud_, pcloudSubSampled_);
    nPoints_=pcloudSubSampled_->size();
  }
  else
  {
    std::vector<int> indVec;
    pcl::removeNaNFromPointCloud(*pcloudOriginal_, *pcloud_, indVec);
    const int nPointsC=750, nPointsR=100;
    int nPointsDiff;
    float seed=2.0e-2, m=0.0, c=5.0e-3;
    do{
      mapPoints(pcloud_);
      resolution_=pointsMap_.row(2).mean()*seed;
      utils::subSamplePCloud<float, pcl::PointXYZ>(pcloud_, resolution_,
        pcloudSubSampled_);
      nPoints_=pcloudSubSampled_->size();

      nPointsDiff=nPoints_-nPointsC;
      if(nPointsDiff<-nPointsR)
      {
        seed-=std::exp(-m)*c;
      }
      else if(nPointsR<nPointsDiff)
      {
        seed+=std::exp(-m)*c;
      }
      m+=10.0*c;
    } while(std::abs(nPointsDiff)>nPointsR);
  }

  accTrackPoints_.setOnes(nPoints_);
  aMean_.setZero(nPoints_);
  meanDist_.setZero(nPoints_, nPoints_), m2Dist_.setZero(nPoints_, nPoints_);
  //c1Mtx_.setZero(nPoints_, nPoints_), c2Mtx_.setZero(nPoints_, nPoints_);
  dDist_.setZero(nPoints_, nPoints_);
  ylMtx_.setOnes(nPoints_, 1);

  nMinPoints_=0.2*nPoints_;
}

void
OnlineKinematicStructure::subSamplePCloud(
  pcl::PointCloud<pcl::PointXYZ>::Ptr& pcloudSubSampled)
{
  utils::subSamplePCloud<float, pcl::PointXYZ>(pcloudOriginal_, nPoints_,
    pcloud_, pcloudSubSampled);
}

void
OnlineKinematicStructure::computeDist(
  const int& opt,
  const int& minEvidence)
{
  //t_.tic();

  computePointsPairwiseDistanceEuclidean();

  Eigen::VectorXf dDistVec(nPoints_);
  Eigen::MatrixXf dDist;

  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octreeSubSampled(resolution_);
  octreeSubSampled.setInputCloud(pcloudSubSampled_);
  octreeSubSampled.addPointsFromInputCloud();
  std::vector<int> indVec;
  std::vector<float> distVec;

  // accumulated variation
  if(opt==0)
  {
    dDist=30.0*(pointsPairwiseDistanceEuclidean_[indToogle_]-pointsPairwiseDistanceEuclidean_[indToogle_^1]).cwiseAbs();
    dDist_.diagonal().setOnes();
    float dDistMin=dDist_.minCoeff();
    while(dDistMin>0.0)
    {
      dDist_.array()-=dDistMin;
      dDistMin=dDist_.minCoeff();
    }      
    dDist_.diagonal().setZero();
    dDist_+=dDist;

    for(int k=0; k<nPoints_; ++k)
    {
      if(accTrackPoints_(k)==1)
      {
        dDistVec.setZero();
        //ylMtx_.row(k).setConstant(1.0/ylMtx_.cols());
        ylMtx_.row(k).setZero();
        //const int naNeighbours=octreeSubSampled.radiusSearch(k, 2.5*resolution_, indVec, distVec);
        const int naNeighbours=octreeSubSampled.nearestKSearch(k, 27, indVec, distVec);
        float influenceNeighbours=0.0;
        for(int i=1; i<naNeighbours; ++i)
        {
          if(accTrackPoints_(indVec[i])>minEvidence)
          {
            const float a=std::exp(-50.0*distVec[i]);
            dDistVec+=a*dDist_.row(indVec[i]);
            //ylMtx_.row(k)+=a*ylMtx_.row(indVec[i]);
            influenceNeighbours+=a;
          }
        }

        if(influenceNeighbours>0.0)
        {
          const float fInfluenceNeighbours=1.0/influenceNeighbours;
          dDistVec*=fInfluenceNeighbours;
          dDistVec(k)=0.0;
          //ylMtx_.row(k)*=fInfluenceNeighbours;
        }
        else
        {
          dDistVec=dDist.row(k);
          //ylMtx_.row(k).setConstant(1.0/ylMtx_.cols());
        }
        dDist_.row(k)=dDistVec;
        dDist_.col(k)=dDist_.row(k).eval();
      }
    }
  }
  // standard variation
  else if(opt==1)
  {
    aMean_.array()+=1.0;
    dDist=30.0*pointsPairwiseDistanceEuclidean_[indToogle_];
    const Eigen::MatrixXf delta1Mtx(dDist-meanDist_);
    for(int k=0; k<nPoints_; ++k)
    {
      meanDist_.row(k)+=(1.0/aMean_(k))*delta1Mtx.row(k);
    }
    m2Dist_+=delta1Mtx.cwiseProduct(dDist-meanDist_);
    for(int k=0; k<nPoints_; ++k)
    {
      dDist_.row(k)=((1.0/aMean_(k))*m2Dist_.row(k)).cwiseSqrt();
    }

    for(int k=0; k<nPoints_; ++k)
    {
      if(accTrackPoints_(k)==1)
      {
        aMean_(k)=0.0;
        dDistVec.setZero();
        //ylMtx_.row(k).setConstant(1.0/ylMtx_.cols());    
        ylMtx_.row(k).setZero();  
        //const int naNeighbours=octreeSubSampled.radiusSearch(k, 2.5*resolution_, indVec, distVec);
        const int naNeighbours=octreeSubSampled.nearestKSearch(k, 27, indVec, distVec);
        float influenceNeighbours=0.0;
        for(int i=0; i<naNeighbours; ++i)
        {
          if(accTrackPoints_(indVec[i])>minEvidence)
          {
            const float a=std::exp(-50.0*distVec[i]);
            dDistVec+=a*m2Dist_.row(indVec[i]);
            //ylMtx_.row(k)+=a*ylMtx_.row(indVec[i]);
            influenceNeighbours+=a;
          }
        }

        if(influenceNeighbours>0.0)
        {
          const float fInfluenceNeighbours=1.0/influenceNeighbours;
          dDistVec*=fInfluenceNeighbours;
          dDistVec(k)=0.0;
          //ylMtx_.row(k)*=(1.0/influenceNeighbours);
        }
        else
        {
          dDistVec=dDist.row(k);
          //ylMtx_.row(k).setConstant(1.0/ylMtx_.cols());
        }
        meanDist_.row(k)=dDist.row(k);
        meanDist_.col(k)=meanDist_.row(k).eval();
        m2Dist_.row(k)=dDistVec;
        m2Dist_.col(k)=m2Dist_.row(k).eval();
      }
    }
  }

  /*if(frame_>1)
  {
    tDist_=t_.toc();
  }*/
}

}
