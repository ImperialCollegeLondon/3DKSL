#include <cmath>
#include <iostream>
#include <ksl/ksl/offline_sf_ksl.h>
#include <ksl/utils/image_io.h>
#include <ksl/utils/tictoc.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <string>
#include <vector>

int
main(int argc, char* argv[])
{
  if(argc!=4)
  {
    std::cout<<"Usage: "<<argv[0]<<" path-to-RGB-dir path-to-Depth-dir number-sub-samples"<<std::endl;
    return -1;
  }

  std::string strargv[argc];
  for(int i=0; i<argc; ++i)
  {
    strargv[i]=argv[i];
  }

  std::vector<cv::Mat> seqRGB, seqDepth;
  const int frames=ksl::utils::loadRGB(strargv[1], seqRGB);
  if(frames<=0)
  {
    std::cout<<"Could not load "<<strargv[1]<<std::endl;
    return -1; 
  }
  if(frames!=ksl::utils::loadDepth(strargv[2], seqDepth))
  {
    std::cout<<"Could not load "<<strargv[2]<<std::endl;
    return -1; 
  }

  ksl::utils::TicToc t;
  pcl::visualization::PCLVisualizer viewer;
  viewer.setBackgroundColor(1.0, 1.0, 1.0);
  const std::string redId="red_points";
  Eigen::MatrixXi cMtx(20, 3);
  cMtx<<219, 209, 0,
        73, 0, 146,
        146, 0, 0,
        0, 146, 146,
        182, 219, 255,
        255, 255, 255,
        128, 0, 0,
        0, 128, 0,
        128, 128, 0,
        128, 255, 0,
        255, 255, 255,
        128, 128, 255,
        255, 128, 128,
        128, 255, 128,
        255, 0, 128,
        0, 255, 128,
        128, 0, 128,
        0, 128, 128,
        64, 0, 0,
        0, 64, 0;

  const float rho=0.65, eigThresh=0.8;
  const int nPoints=atoi(argv[3]);
  const int nMaxClusters=std::ceil(std::log2(nPoints));
  ksl::OfflineKinematicStructure ksl(
    seqRGB[0].cols, seqRGB[0].rows, seqRGB[0].rows, frames, nMaxClusters, nPoints, rho, eigThresh);
  //const float resolution=atof(argv[3]);
  //ksl::OfflineKinematicStructure ksl(
    //seqRGB[0].cols, seqRGB[0].rows, seqRGB[0].rows, frames, nMaxClusters, resolution, rho, eigThresh);

  for(int f=1; f<frames; ++f)
  {
    viewer.removeAllPointClouds();

    t.tic();
    ksl.computeSceneFlow(seqRGB[f-1], seqRGB[f], seqDepth[f-1], seqDepth[f]);
    t.toc("computeSceneFlow(): ");

    t.tic();
    ksl.trackPoints(seqRGB[f], seqDepth[f], 5.0e-3);
    t.toc("trackPoints(): ");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      redColor(ksl.pcloudSubSampled(), 255, 0, 0);
    viewer.addPointCloud<pcl::PointXYZ>(ksl.pcloudSubSampled(), redColor, redId);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, redId);
    while(t.toc()<=0.2)
    {
      viewer.spinOnce();
    }
  }

  t.tic();
  ksl.computeKSL();
  t.toc("computeKSL(): ");
  std::cout<<"nGroups: "<<ksl.nGroups()<<std::endl;

  viewer.removeAllPointClouds();
  std::string sId, sphereId;
  for(int i=0; i<ksl.nGroups(); ++i)
  {
    sId="sId"+std::to_string(i);
    sphereId="sphereId"+std::to_string(i);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      ptColor(ksl.pcloudSubSampled(), cMtx(i, 0), cMtx(i, 1), cMtx(i, 2));
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pt(0.0, 0.0, 0.0);
    int nPt=0;
    for(int j=0; j<ksl.groups().size(); ++j)
    {
      if(ksl.groups()(j)==i)
      {
        pt.x+=ksl.pcloudSubSampled()->at(j).x;
        pt.y+=ksl.pcloudSubSampled()->at(j).y;
        pt.z+=ksl.pcloudSubSampled()->at(j).z;
        ++nPt;
        pcloud->push_back(ksl.pcloudSubSampled()->at(j));
      }
    }
    viewer.addPointCloud<pcl::PointXYZ>(pcloud, ptColor, sId);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, sId); 
    pt.x/=nPt;
    pt.y/=nPt;
    pt.z/=nPt;
    viewer.addSphere(pt, 0.01, 0.0, 0.0, 0.0, sphereId);
  }

  while(!viewer.wasStopped())
  {
    viewer.spinOnce();
  }

  return 0;
}
