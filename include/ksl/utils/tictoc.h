#ifndef TICTOC_H
#define TICTOC_H

#include <chrono>
#include <ctime>
#include <iostream>
#include <string>

namespace ksl
{

namespace utils
{

class TicToc
{

private:

protected:

  std::chrono::high_resolution_clock::time_point t_;

public:

  TicToc(void)
  {}
  ~TicToc(void)
  {}

  inline void
  tic(void)
  {
    t_=std::chrono::high_resolution_clock::now();
  }
  inline double
  toc(
    const int msecOpt=0)
  {
    std::chrono::high_resolution_clock::time_point t=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT=std::chrono::duration_cast<std::chrono::duration<double> >(t-t_);
    const double dT=deltaT.count();
    if(msecOpt==0)
    {
      return dT;
    }
    else if(msecOpt==1)
    {
      return dT*1000.0;
    }
    return -1.0;
  }
  inline void
  toc(
    const std::string& msg,
    const int msecOpt=0)
  {
    std::chrono::high_resolution_clock::time_point t=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT=std::chrono::duration_cast<std::chrono::duration<double> >(t-t_);
    const double dT=deltaT.count();
    std::cout<<msg;
    if(msecOpt==0)
    {
      std::cout<<dT<<" [sec]";
    }
    else if(msecOpt==1)
    {
      std::cout<<dT*1000.0<<" [msec]";
    }
    std::cout<<std::endl;
  }

protected:

private:

};

}

}

#endif // TICTOC_H
