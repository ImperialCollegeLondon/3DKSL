#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <dirent.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace ksl
{

namespace utils
{

template<typename T, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
inline void
eigen2cv(
  const Eigen::Matrix<T, _rows, _cols, _options, _maxRows, _maxCols>& src,
  cv::Mat& dst)
{
  if(!(src.Flags & Eigen::RowMajorBit))
  {
    cv::Mat _src(src.cols(), src.rows(), cv::traits::Type<T>::value,
      (void*) src.data(), src.stride()*sizeof(T));
    cv::transpose(_src, dst);
  }
  else
  {
    cv::Mat _src(src.rows(), src.cols(), cv::traits::Type<T>::value,
      (void*) src.data(), src.stride()*sizeof(T));
    _src.copyTo(dst);
  }
}

bool
loadRGB(
  const std::string& pathDir,
  cv::Mat& img)
{
  img=cv::imread(pathDir, CV_LOAD_IMAGE_COLOR);
  if(img.empty())
  {
    return false;
  }
  return true;
}

bool
loadGray(
  const std::string& pathDir,
  cv::Mat& img)
{
  img=cv::imread(pathDir, CV_LOAD_IMAGE_GRAYSCALE);
  if(img.empty())
  {
    return false;
  }
  return true;
}

bool
loadDepth(
  const std::string& pathDir,
  cv::Mat& img,
  const float& scale=1.0/5000.0)
{
  img=cv::imread(pathDir, CV_LOAD_IMAGE_ANYDEPTH);
  if(img.empty())
  {
    return false;
  }
  img.convertTo(img, CV_32FC1, scale);
  return true;
}

int
loadRGB(
  const std::string& pathDir,
  std::vector<cv::Mat>& seqImg,
  const int& maxFiles=0)
{
  struct dirent **fileNameList;
  int nFiles=scandir(pathDir.c_str(), &fileNameList, NULL, alphasort);
  if(nFiles<0)
  {
    return 0;
  }
  if(nFiles>maxFiles && maxFiles>0)
  {
    nFiles=maxFiles+2;
  }

  seqImg.resize(0);
  struct stat fstat;
  std::string fileName;
  cv::Mat img;
  int frames=nFiles;
  for(int f=0; f<nFiles; ++f)
  {
    fileName=fileNameList[f]->d_name;
    fileName=pathDir+"/"+fileName;
    /* check for directories or invalid files */
    if(stat(fileName.c_str(), &fstat)==-1)
    {
      --frames;
    }
    else if(S_ISDIR(fstat.st_mode))
    {
      --frames;
    }
    else
    {
      if(loadRGB(fileName, img))
      {
        seqImg.push_back(img);
      }
      else
      {
        --frames;
      }
    }
    free(fileNameList[f]);
  }
  free(fileNameList);

  return frames;
}

int
loadGray(
  const std::string& pathDir,
  std::vector<cv::Mat>& seqImg,
  const int& maxFiles=0)
{
  struct dirent **fileNameList;
  int nFiles=scandir(pathDir.c_str(), &fileNameList, NULL, alphasort);
  if(nFiles<0)
  {
    return 0;
  }
  if(nFiles>maxFiles && maxFiles>0)
  {
    nFiles=maxFiles+2;
  }

  seqImg.resize(0);
  struct stat fstat;
  std::string fileName;
  cv::Mat img;
  int frames=nFiles;
  for(int f=0; f<nFiles; ++f)
  {
    fileName=fileNameList[f]->d_name;
    fileName=pathDir+"/"+fileName;
    /* check for directories or invalid files */
    if(stat(fileName.c_str(), &fstat)==-1)
    {
      --frames;
    }
    else if(S_ISDIR(fstat.st_mode))
    {
      --frames;
    }
    else
    {
      if(loadGray(fileName, img))
      {
        seqImg.push_back(img);
      }
      else
      {
        --frames;
      }
    }
    free(fileNameList[f]);
  }
  free(fileNameList);

  return frames;
}

int
loadDepth(
  const std::string& pathDir,
  std::vector<cv::Mat>& seqImg,
  const int& maxFiles=0,
  const float& scale=1.0/5000.0)
{
  struct dirent **fileNameList;
  int nFiles=scandir(pathDir.c_str(), &fileNameList, NULL, alphasort);
  if(nFiles<0)
  {
    return 0;
  }
  if(nFiles>maxFiles && maxFiles>0)
  {
    nFiles=maxFiles+2;
  }

  seqImg.resize(0);
  struct stat fstat;
  std::string fileName;
  cv::Mat img;
  int frames=nFiles;
  //double maxVal=-1;
  for(int f=0; f<nFiles; ++f)
  {
    fileName=fileNameList[f]->d_name;
    fileName=pathDir+"/"+fileName;
    /* check for directories or invalid files */
    if(stat(fileName.c_str(), &fstat)==-1)
    {
      --frames;
    }
    else if(S_ISDIR(fstat.st_mode))
    {
      --frames;
    }
    else
    {
      if(loadDepth(fileName, img, scale))
      {
        /*double min, max;
        cv::minMaxLoc(img, &min, &max);
        if(maxVal<max)
        {
          maxVal=max;
        }*/
        seqImg.push_back(img);
      }
      else
      {
        --frames;
      }
    }
    free(fileNameList[f]);
  }
  free(fileNameList);

  /*for(int f=0; f<frames; f++)
  {
    seqImg[f].convertTo(seqImg[f], CV_32FC1, 1.0/maxVal);
  }*/

  return frames;
}

}

}

#endif // IMAGE_IO_H
