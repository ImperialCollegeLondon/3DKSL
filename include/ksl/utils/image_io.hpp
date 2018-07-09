#ifndef IMAGE_IO_HPP
#define IMAGE_IO_HPP

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

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline void
eigen2cv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> &src, cv::Mat &dst)
{
  if(!(src.Flags & Eigen::RowMajorBit))
  {
    cv::Mat _src(src.cols(), src.rows(), cv::traits::Type<_Tp>::value,
      (void*) src.data(), src.stride()*sizeof(_Tp));
    cv::transpose(_src, dst);
  }
  else
  {
    cv::Mat _src(src.rows(), src.cols(), cv::traits::Type<_Tp>::value,
      (void*) src.data(), src.stride()*sizeof(_Tp));
    _src.copyTo(dst);
  }
}

int
loadRGB(
  const std::string &pathDir,
  cv::Mat &img)
{
  img=cv::imread(pathDir, CV_LOAD_IMAGE_COLOR);
  if(img.empty())
  {
    return 0;
  }
  return 1;
}

int
loadGray(
  const std::string &pathDir,
  cv::Mat &img)
{
  img=cv::imread(pathDir, CV_LOAD_IMAGE_GRAYSCALE);
  if(img.empty())
  {
    return 0;
  }
  return 1;
}

int
loadDepth(
  const std::string &pathDir,
  cv::Mat &img)
{
  img=cv::imread(pathDir, CV_LOAD_IMAGE_ANYDEPTH);
  if(img.empty())
  {
    return 0;
  }
  img.convertTo(img, CV_32FC1, 1/1000.0);
  return 1;
}

int
loadRGB(
  const std::string &pathDir,
  std::vector<cv::Mat> &seqImg)
{
  struct dirent **fileNameList;
  int nFiles=scandir(pathDir.c_str(), &fileNameList, NULL, alphasort);
  if(nFiles<0)
  {
    return 0;
  }

  seqImg.resize(0);
  struct stat fstat;
  std::string fileName;
  cv::Mat img;
  int frames=nFiles;
  for(int f=0; f<nFiles; f++)
  {
    fileName=fileNameList[f]->d_name;
    fileName=pathDir+"/"+fileName;
    /* check for directories or invalid files */
    if(stat(fileName.c_str(), &fstat)==-1)
    {
      frames--;
    }
    else if(S_ISDIR(fstat.st_mode))
    {
      frames--;
    }
    else
    {
      if(loadRGB(fileName, img))
      {
        seqImg.push_back(img);
      }
      else
      {
        frames--;
      }
    }
    free(fileNameList[f]);
  }
  free(fileNameList);

  return frames;
}

int
loadGray(
  const std::string &pathDir,
  std::vector<cv::Mat> &seqImg)
{
  struct dirent **fileNameList;
  int nFiles=scandir(pathDir.c_str(), &fileNameList, NULL, alphasort);
  if(nFiles<0)
  {
    return 0;
  }

  seqImg.resize(0);
  struct stat fstat;
  std::string fileName;
  cv::Mat img;
  int frames=nFiles;
  for(int f=0; f<nFiles; f++)
  {
    fileName=fileNameList[f]->d_name;
    fileName=pathDir+"/"+fileName;
    /* check for directories or invalid files */
    if(stat(fileName.c_str(), &fstat)==-1)
    {
      frames--;
    }
    else if(S_ISDIR(fstat.st_mode))
    {
      frames--;
    }
    else
    {
      if(loadGray(fileName, img))
      {
        seqImg.push_back(img);
      }
      else
      {
        frames--;
      }
    }
    free(fileNameList[f]);
  }
  free(fileNameList);

  return frames;
}

int
loadDepth(
  const std::string &pathDir,
  std::vector<cv::Mat> &seqImg)
{
  struct dirent **fileNameList;
  int nFiles=scandir(pathDir.c_str(), &fileNameList, NULL, alphasort);
  if(nFiles<0)
  {
    return 0;
  }

  seqImg.resize(0);
  struct stat fstat;
  std::string fileName;
  cv::Mat img;
  int frames=nFiles;
  for(int f=0; f<nFiles; f++)
  {
    fileName=fileNameList[f]->d_name;
    fileName=pathDir+"/"+fileName;
    /* check for directories or invalid files */
    if(stat(fileName.c_str(), &fstat)==-1)
    {
      frames--;
    }
    else if(S_ISDIR(fstat.st_mode))
    {
      frames--;
    }
    else
    {
      if(loadDepth(fileName, img))
      {
        /*double min, max;
        cv::minMaxLoc(img, &min, &max);
        std::cout<<max<<std::endl;*/
        seqImg.push_back(img);
      }
      else
      {
        frames--;
      }
    }
    free(fileNameList[f]);
  }
  free(fileNameList);

  return frames;
}

}

}

#endif // IMAGE_IO_HPP

