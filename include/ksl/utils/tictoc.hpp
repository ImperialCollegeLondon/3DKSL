#ifndef TICTOC_HPP
#define TICTOC_HPP

#include <iostream>
#include <string>
#include <time.h>

namespace ksl
{

namespace utils
{

class TicToc
{

private:

protected:

  clock_t t_;

public:

  TicToc(void): t_(0)
  {}
  ~TicToc(void)
  {}

  inline void
  tic(void)
  {
    t_=clock();
  }
  inline float
  toc(
    const int msecOpt=0)
  {
    float t=((float) clock()-t_)/CLOCKS_PER_SEC;
    if(msecOpt==0)
    {
      return t;
    }
    else if(msecOpt==1)
    {
      return t*1000.0;
    }
    return -1.0;
  }
  inline void
  toc(
    const std::string &msg,
    const int msecOpt=0)
  {
    float t=((float) clock()-t_)/CLOCKS_PER_SEC;
    std::cout<<msg;
    if(msecOpt==0)
    {
      std::cout<<t<<" [sec]";
    }
    else if(msecOpt==1)
    {
      t*=1000.0;
      std::cout<<t<<" [msec]";
    }
    std::cout<<std::endl;
  }

protected:

private:

};

}

}

#endif // TICTOC_HPP

