#include <cstdlib>
#include <iostream>
#include <ksl/ksl/online_sf_ksl.h>
#include <ksl/utils/image_io.h>
#include <ksl/utils/tictoc.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <string>
#include <vector>

int
main(int argc, char *argv[])
{
  if(argc<3)
  {
    std::cout<<"Usage: "<<argv[0]<<" path-to-RGB-dir path-to-Depth-dir"<<std::endl;
    return -1;
  }

  std::string strargv[argc];
  for(int i=0; i<argc; i++)
  {
    strargv[i]=argv[i];
  }

  const int maxFiles=500;
  std::vector<cv::Mat> seqRGB, seqDepth;
  const int frames=ksl::utils::loadRGB(strargv[1], seqRGB, maxFiles);
  if(frames<=0)
  {
    std::cout<<"Could not load "<<strargv[1]<<std::endl;
    return -1; 
  }
  if(frames!=ksl::utils::loadDepth(strargv[2], seqDepth, maxFiles, 1.0/1000.0))
  {
    std::cout<<"Could not load "<<strargv[2]<<std::endl;
    return -1; 
  }

  ksl::utils::TicToc t, tTotal;
  const std::string pointsDistanceId("dDistance"), accAdjacencyId("accAdjacency");
  cv::Mat pointsDistanceImg, accAdjacencyImg;

  const std::string redId("red_points");
  //const std::string newPointsId("new_points");
  pcl::visualization::PCLVisualizer viewer;
  viewer.setBackgroundColor(1.0, 1.0, 1.0);
  Eigen::MatrixXi cMtx(20, 3);
  cMtx<<219, 209, 0,
        73, 0, 146,
        146, 0, 0,
        0, 146, 146,
        128, 255, 0,
        128, 128, 128,
        182, 219, 255,
        0, 0, 128,
        128, 0, 0,
        0, 128, 0,
        128, 128, 0,
        128, 128, 255,
        255, 128, 128,
        128, 255, 128,
        255, 0, 128,
        0, 255, 128,
        128, 0, 128,
        0, 128, 128,
        64, 0, 0,
        0, 64, 0;

  const ksl::utils::CamParams<float> camParams(seqRGB[0].cols, seqRGB[0].rows);
  //ksl::utils::CamParams<float> camParams(seqRGB[0].cols, seqRGB[0].rows, 480.0, 270.0, 564.3, 564.3);
  //ksl::utils::CamParams<float> camParams(seqRGB[0].cols, seqRGB[0].rows, 328.9866333007812, 237.7507629394531, 618.3587036132812, 618.5924072265625);
  //ksl::utils::CamParams<float> camParams(seqRGB[0].cols, seqRGB[0].rows, 350.5420132033752, 237.5509106406638, 575.8492787648053, 569.2967821532719);

  const float scale=(argc>=4) ? std::atof(argv[3]) : 2.0;
  const int opt=(argc>=5) ? std::atoi(argv[4]) : 0, minEvidence=30/2;
  const float beta=(argc>=6) ? std::atof(argv[5]) : 0.275;

  const int nEig=(argc>=7) ? std::atoi(argv[6]) : 3;
  const bool useNeg=(argc>=8) ? std::atoi(argv[7]) : false;
  const bool remNoise=(argc>=9) ? std::atoi(argv[8]) : true;
  const ksl::affinity::NonMetricTwoViewsParams<float> aParams(nEig, 0.01, 1.0e-6, useNeg, remNoise);
  
  const float alpha=(argc>=10) ? std::atof(argv[9]) : 0.75;
  const float tau=(argc>=11) ? std::atof(argv[10]) : (opt==0) ? 2.5e-2 : 1.0e-3;
  const float fracSplit=(argc>=12) ? std::atof(argv[11]) : (opt==0) ? 0.025 : 0.0425;
  const ksl::clustering::LabelPropagationParams<float> cParams(alpha, tau, 100.0, 1.0e-6, 1.0e-3, fracSplit, 500, 2);

  ksl::OnlineKinematicStructure ksl(
    seqRGB[0].cols, seqRGB[0].rows, scale,
    camParams, beta,
    aParams, cParams);
  /*const int nPoints=1000;
  ksl::OnlineKinematicStructure ksl(
    seqRGB[0].cols, seqRGB[0].rows, scale,
    camParams, beta,
    aParams, cParams, nPoints);*/
  /*const float resolution=0.025;
  ksl::OnlineKinematicStructure ksl(
    seqRGB[0].cols, seqRGB[0].rows, scale,
    camParams, beta,
    aParams, cParams, resolution);*/

  for(int f=1; f<frames; ++f)
  {
    viewer.removeAllPointClouds();
    viewer.removeAllShapes();
    std::cout<<"frame "<<f+1<<"/"<<frames<<std::endl;

    t.tic();
    ksl.computeSceneFlow(seqRGB[f-1], seqRGB[f], seqDepth[f-1], seqDepth[f]);
    t.toc("computeSceneFlow(): ");

    tTotal.tic();
    t.tic();
    ksl.trackPoints(seqRGB[f], seqDepth[f], minEvidence, 5.0e-3);
    t.toc("trackPoints(): ");
    std::cout<<"nLostPoints: "<<ksl.nLostPoints()<<std::endl;

    t.tic();
    ksl.computeKSL(opt, minEvidence);
    t.toc("computeKSL(): ");
    std::cout<<"nGroups: "<<ksl.nGroups()<<std::endl;
    tTotal.toc("  total time: ");

    //pcl::PointCloud<pcl::PointXYZ>::Ptr pcloudNew(new pcl::PointCloud<pcl::PointXYZ>);
    std::string sId, sphereId;
    Eigen::MatrixXf centroids(ksl.nGroups(), 4);
    for(int i=0; i<ksl.nGroups(); ++i)
    {
      sId="sId"+std::to_string(i);
      sphereId="sphereId"+std::to_string(i);
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud(new pcl::PointCloud<pcl::PointXYZ>);
      int countPoints=0, k=0;
      for(int j=0; j<ksl.nPoints(); ++j)
      {
        if(ksl.accTrackPoints()(j)>minEvidence)
        {
          if(ksl.groups()(k)==i)
          {
            pcl::PointXYZ pt=ksl.pcloudSubSampled()->at(j);
            pt.z=-pt.z;
            pcloud->push_back(pt);
            ++countPoints;
          }
          ++k;
        }
        //else
        //{
          //pcloudNew->push_back(ksl.pcloudSubSampled()->at(j));
        //}
      }
      std::cout<<i<<": "<<countPoints<<std::endl;
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        ptColor(pcloud, cMtx(i, 0), cMtx(i, 1), cMtx(i, 2));
      viewer.addPointCloud<pcl::PointXYZ>(pcloud, ptColor, sId);
      viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, sId);
      
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*pcloud, centroid);
      centroids.row(i)=centroid;
      viewer.addSphere(pcl::PointXYZ(centroid(0), centroid(1), centroid(2)), 0.02,
        0.0, 0.0, 0.0, sphereId);
    }
    for(int i=0; i<ksl.nKs(); ++i)
    {
      const std::string lineId("lineId"+std::to_string(i));
      const int ii=ksl.ks()(i, 0), jj=ksl.ks()(i, 1);
      viewer.addLine<pcl::PointXYZ>(pcl::PointXYZ(centroids(ii, 0), centroids(ii, 1), centroids(ii, 2)),
        pcl::PointXYZ(centroids(jj, 0), centroids(jj, 1), centroids(jj, 2)),
        1.0, 0.0, 0.0, lineId);
      viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, lineId);
    }
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      //ptColor(ksl.pcloudSubSampled(), 0, 0, 0);
    //viewer.addPointCloud<pcl::PointXYZ>(ksl.pcloudSubSampled(), ptColor, newPointsId);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      //ptColor(pcloudNew, 0, 0, 0);
    //viewer.addPointCloud<pcl::PointXYZ>(pcloudNew, ptColor, newPointsId);
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, newPointsId);
    viewer.resetCamera();
    viewer.spinOnce(1);

    const Eigen::MatrixXf dDist(10.0*ksl.dDist());
    ksl::utils::eigen2cv(dDist, pointsDistanceImg);
    cv::imshow(pointsDistanceId, pointsDistanceImg);

    if(ksl.nGroups()>0)
    {
      const Eigen::MatrixXf accAdjacency(10.0*ksl.accAdjacency());
      ksl::utils::eigen2cv(accAdjacency, accAdjacencyImg);
      cv::imshow(accAdjacencyId, accAdjacencyImg);
    }

    cv::waitKey(10);
  }

  while(!viewer.wasStopped())
  {
    viewer.spinOnce();
  }

  return 0;
}
