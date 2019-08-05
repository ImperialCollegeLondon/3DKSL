#include <ksl/clustering/hungarian.h>

/*
Copyright (c) 2004, Markus Buehren
All rights reserved.
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:
    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright 
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the distribution
      
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

namespace ksl
{

namespace clustering
{

#define HUNGARIAN_CHECK_FOR_INF
//#define HUNGARIAN_ONE_INDEXING

template<typename T>
static inline void assignmentoptimal(T *assignment, T *cost, const T *distMatrixIn, int nOfRows, int nOfColumns);
template<typename T>
static inline void buildassignmentvector(T *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
template<typename T>
static inline void computeassignmentcost(T *assignment, T *cost, const T *distMatrix, int nOfRows);
template<typename T>
static inline void step2a(T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
template<typename T>
static inline void step2b(T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
template<typename T>
static inline void step3 (T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
template<typename T>
static inline void step4 (T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
template<typename T>
static inline void step5 (T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);

template<typename T>
void assignmentoptimal(
  T *assignment, 
  T *cost, 
  const T *distMatrixIn, 
  int nOfRows, 
  int nOfColumns)
{
    T *distMatrix, *distMatrixTemp, *distMatrixEnd, *columnEnd, value, minValue;
    bool *coveredColumns, *coveredRows, *starMatrix, *newStarMatrix, *primeMatrix;
    int nOfElements, minDim, row, col;
#ifdef HUNGARIAN_CHECK_FOR_INF
    bool infiniteValueFound;
    T maxFiniteValue, infValue;
#endif
    
    /* initialization */
    *cost = 0;
    for(row=0; row<nOfRows; row++)
#ifdef HUNGARIAN_ONE_INDEXING
        assignment[row] =  0.0;
#else
        assignment[row] = -1.0;
#endif
    
    /* generate working copy of distance Matrix */
    /* check if all matrix elements are positive */
    nOfElements   = nOfRows * nOfColumns;
    //distMatrix    = (T *)mxMalloc(nOfElements * sizeof(T));
        distMatrix = new T[nOfElements];
    distMatrixEnd = distMatrix + nOfElements;
    for(row=0; row<nOfElements; row++)
    {
        value = distMatrixIn[row];
        assert(!(std::isfinite(value) && (value < 0)) &&
          "All matrix elements have to be non-negative.");
        distMatrix[row] = value;
    }

#ifdef HUNGARIAN_CHECK_FOR_INF
    /* check for infinite values */
    maxFiniteValue     = -1;
    infiniteValueFound = false;
    
    distMatrixTemp = distMatrix;
    while(distMatrixTemp < distMatrixEnd)
    {
        value = *distMatrixTemp++;
        if(std::isfinite(value))
        {
            if(value > maxFiniteValue)
                maxFiniteValue = value;
        }
        else
            infiniteValueFound = true;
    }
    if(infiniteValueFound)
    {
        if(maxFiniteValue == -1) /* all elements are infinite */
            return;
        
        /* set all infinite elements to big finite value */
        if(maxFiniteValue > 0)
            infValue = 10 * maxFiniteValue * nOfElements;
        else
            infValue = 10;
        distMatrixTemp = distMatrix;
        while(distMatrixTemp < distMatrixEnd)
            if(std::isinf(*distMatrixTemp++))
                *(distMatrixTemp-1) = infValue;
    }
#endif
                
    /* memory allocation */
    coveredColumns = (bool *)calloc(nOfColumns,  sizeof(bool));
    coveredRows    = (bool *)calloc(nOfRows,     sizeof(bool));
    starMatrix     = (bool *)calloc(nOfElements, sizeof(bool));
    primeMatrix    = (bool *)calloc(nOfElements, sizeof(bool));
    newStarMatrix  = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

    /* preliminary steps */
    if(nOfRows <= nOfColumns)
    {
        minDim = nOfRows;
        
        for(row=0; row<nOfRows; row++)
        {
            /* find the smallest element in the row */
            distMatrixTemp = distMatrix + row;
            minValue = *distMatrixTemp;
            distMatrixTemp += nOfRows;          
            while(distMatrixTemp < distMatrixEnd)
            {
                value = *distMatrixTemp;
                if(value < minValue)
                    minValue = value;
                distMatrixTemp += nOfRows;
            }
            
            /* subtract the smallest element from each element of the row */
            distMatrixTemp = distMatrix + row;
            while(distMatrixTemp < distMatrixEnd)
            {
                *distMatrixTemp -= minValue;
                distMatrixTemp += nOfRows;
            }
        }
        
        /* Steps 1 and 2a */
        for(row=0; row<nOfRows; row++)
            for(col=0; col<nOfColumns; col++)
                if(distMatrix[row + nOfRows*col] == 0)
                    if(!coveredColumns[col])
                    {
                        starMatrix[row + nOfRows*col] = true;
                        coveredColumns[col]           = true;
                        break;
                    }
    }
    else /* if(nOfRows > nOfColumns) */
    {
        minDim = nOfColumns;
        
        for(col=0; col<nOfColumns; col++)
        {
            /* find the smallest element in the column */
            distMatrixTemp = distMatrix     + nOfRows*col;
            columnEnd      = distMatrixTemp + nOfRows;
            
            minValue = *distMatrixTemp++;           
            while(distMatrixTemp < columnEnd)
            {
                value = *distMatrixTemp++;
                if(value < minValue)
                    minValue = value;
            }
            
            /* subtract the smallest element from each element of the column */
            distMatrixTemp = distMatrix + nOfRows*col;
            while(distMatrixTemp < columnEnd)
                *distMatrixTemp++ -= minValue;
        }
        
        /* Steps 1 and 2a */
        for(col=0; col<nOfColumns; col++)
            for(row=0; row<nOfRows; row++)
                if(distMatrix[row + nOfRows*col] == 0)
                    if(!coveredRows[row])
                    {
                        starMatrix[row + nOfRows*col] = true;
                        coveredColumns[col]           = true;
                        coveredRows[row]              = true;
                        break;
                    }
        for(row=0; row<nOfRows; row++)
            coveredRows[row] = false;
        
    }   
    
    /* move to step 2b */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

    /* compute cost and remove invalid assignments */
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
    
    /* free allocated memory */
    free(distMatrix);
    free(coveredColumns);
    free(coveredRows);
    free(starMatrix);
    free(primeMatrix);
    free(newStarMatrix);

    return;
}

/********************************************************/
template<typename T>
void buildassignmentvector(T *assignment, bool *starMatrix, int nOfRows, int nOfColumns)
{
    int row, col;
    
    for(row=0; row<nOfRows; row++)
        for(col=0; col<nOfColumns; col++)
            if(starMatrix[row + nOfRows*col])
            {
#ifdef HUNGARIAN_ONE_INDEXING
                assignment[row] = col + 1; /* MATLAB-Indexing */
#else
                assignment[row] = col;
#endif
                break;
            }
}

/********************************************************/
template<typename T>
void computeassignmentcost(T *assignment, T *cost, const T *distMatrix, int nOfRows)
{
    int row, col;
#ifdef HUNGARIAN_CHECK_FOR_INF
    T value;
#endif
    
    for(row=0; row<nOfRows; row++)
    {
#ifdef HUNGARIAN_ONE_INDEXING
        col = assignment[row]-1; /* MATLAB-Indexing */
#else
        col = assignment[row];
#endif

        if(col >= 0)
        {
#ifdef HUNGARIAN_CHECK_FOR_INF
            value = distMatrix[row + nOfRows*col];
            if(std::isfinite(value))
                *cost += value;
            else
#ifdef HUNGARIAN_ONE_INDEXING
                assignment[row] =  0.0;
#else
                assignment[row] = -1.0;
#endif

#else
            *cost += distMatrix[row + nOfRows*col];
#endif
        }
    }
}

/********************************************************/
template<typename T>
void step2a(T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    bool *starMatrixTemp, *columnEnd;
    int col;
    
    /* cover every column containing a starred zero */
    for(col=0; col<nOfColumns; col++)
    {
        starMatrixTemp = starMatrix     + nOfRows*col;
        columnEnd      = starMatrixTemp + nOfRows;
        while(starMatrixTemp < columnEnd){
            if(*starMatrixTemp++)
            {
                coveredColumns[col] = true;
                break;
            }
        }   
    }

    /* move to step 3 */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
template<typename T>
void step2b(T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    int col, nOfCoveredColumns;
    
    /* count covered columns */
    nOfCoveredColumns = 0;
    for(col=0; col<nOfColumns; col++)
        if(coveredColumns[col])
            nOfCoveredColumns++;
            
    if(nOfCoveredColumns == minDim)
    {
        /* algorithm finished */
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
    else
    {
        /* move to step 3 */
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
    
}

/********************************************************/
template<typename T>
void step3(T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    bool zerosFound;
    int row, col, starCol;

    zerosFound = true;
    while(zerosFound)
    {
        zerosFound = false;     
        for(col=0; col<nOfColumns; col++)
            if(!coveredColumns[col])
                for(row=0; row<nOfRows; row++)
                    if((!coveredRows[row]) && (distMatrix[row + nOfRows*col] == 0))
                    {
                        /* prime zero */
                        primeMatrix[row + nOfRows*col] = true;
                        
                        /* find starred zero in current row */
                        for(starCol=0; starCol<nOfColumns; starCol++)
                            if(starMatrix[row + nOfRows*starCol])
                                break;
                        
                        if(starCol == nOfColumns) /* no starred zero found */
                        {
                            /* move to step 4 */
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        else
                        {
                            coveredRows[row]        = true;
                            coveredColumns[starCol] = false;
                            zerosFound              = true;
                            break;
                        }
                    }
    }
    
    /* move to step 5 */
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
template<typename T>
void step4(T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{   
    int n, starRow, starCol, primeRow, primeCol;
    int nOfElements = nOfRows*nOfColumns;
    
    /* generate temporary copy of starMatrix */
    for(n=0; n<nOfElements; n++)
        newStarMatrix[n] = starMatrix[n];
    
    /* star current zero */
    newStarMatrix[row + nOfRows*col] = true;

    /* find starred zero in current column */
    starCol = col;
    for(starRow=0; starRow<nOfRows; starRow++)
        if(starMatrix[starRow + nOfRows*starCol])
            break;

    while(starRow<nOfRows)
    {
        /* unstar the starred zero */
        newStarMatrix[starRow + nOfRows*starCol] = false;
    
        /* find primed zero in current row */
        primeRow = starRow;
        for(primeCol=0; primeCol<nOfColumns; primeCol++)
            if(primeMatrix[primeRow + nOfRows*primeCol])
                break;
                                
        /* star the primed zero */
        newStarMatrix[primeRow + nOfRows*primeCol] = true;
    
        /* find starred zero in current column */
        starCol = primeCol;
        for(starRow=0; starRow<nOfRows; starRow++)
            if(starMatrix[starRow + nOfRows*starCol])
                break;
    }   

    /* use temporary copy as new starMatrix */
    /* delete all primes, uncover all rows */
    for(n=0; n<nOfElements; n++)
    {
        primeMatrix[n] = false;
        starMatrix[n]  = newStarMatrix[n];
    }
    for(n=0; n<nOfRows; n++)
        coveredRows[n] = false;
    
    /* move to step 2a */
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
template<typename T>
void step5(T *assignment, T *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    T h, value;
    int row, col;
    
    /* find smallest uncovered element h */
    h = std::numeric_limits<T>::infinity();
    for(row=0; row<nOfRows; row++)
        if(!coveredRows[row])
            for(col=0; col<nOfColumns; col++)
                if(!coveredColumns[col])
                {
                    value = distMatrix[row + nOfRows*col];
                    if(value < h)
                        h = value;
                }
    
    /* add h to each covered row */
    for(row=0; row<nOfRows; row++)
        if(coveredRows[row])
            for(col=0; col<nOfColumns; col++)
                distMatrix[row + nOfRows*col] += h;
    
    /* subtract h from each uncovered column */
    for(col=0; col<nOfColumns; col++)
        if(!coveredColumns[col])
            for(row=0; row<nOfRows; row++)
                distMatrix[row + nOfRows*col] -= h;
    
    /* move to step 3 */
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}


template <typename DerivedD, typename DerivedA, typename c_type> 
void hungarian(
  const Eigen::PlainObjectBase<DerivedD> & D,
  Eigen::PlainObjectBase<DerivedA> & A,
  c_type & c)
{
  A.resize(D.rows(),1);
  Eigen::Matrix<c_type, Eigen::Dynamic, 1> Ad(A.rows());
  assignmentoptimal<c_type>(Ad.data(),&c,D.data(),D.rows(),D.cols());
  A = Ad.template cast<int>();
}

// Explicit template instanciation
template void assignmentoptimal<double>(double *, double *, const double *, int , int );
template void assignmentoptimal<float>(float *, float *, const float *, int , int );

template void buildassignmentvector<double>(double *, bool *, int , int );
template void buildassignmentvector<float>(float *, bool *, int , int );

template void computeassignmentcost<double>(double *, double *, const double *, int );
template void computeassignmentcost<float>(float *, float *, const float *, int );

template void step2a<double>(double *, double *, bool *, bool *, bool *, bool *, bool *, int , int , int );
template void step2a<float>(float *, float *, bool *, bool *, bool *, bool *, bool *, int , int , int );

template void step2b<double>(double *, double *, bool *, bool *, bool *, bool *, bool *, int , int , int );
template void step2b<float>(float *, float *, bool *, bool *, bool *, bool *, bool *, int , int , int );

template void step3<double>(double *, double *, bool *, bool *, bool *, bool *, bool *, int , int , int );
template void step3<float>(float *, float *, bool *, bool *, bool *, bool *, bool *, int , int , int );

template void step4<double>(double *, double *, bool *, bool *, bool *, bool *, bool *, int , int , int , int , int );
template void step4<float>(float *, float *, bool *, bool *, bool *, bool *, bool *, int , int , int , int , int );

template void step5<double>(double *, double *, bool *, bool *, bool *, bool *, bool *, int , int , int );
template void step5<float>(float *, float *, bool *, bool *, bool *, bool *, bool *, int , int , int );

template void hungarian<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<int, Eigen::Dynamic, 1>, double>(Eigen::PlainObjectBase<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, Eigen::Dynamic, 1> >&, double&);
template void hungarian<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<int, Eigen::Dynamic, 1>, float>(Eigen::PlainObjectBase<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, Eigen::Dynamic, 1> >&, float&);

}

}
