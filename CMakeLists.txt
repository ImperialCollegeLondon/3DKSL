cmake_minimum_required(VERSION 3.0)

set(LIB_NAME ksl)
project(${LIB_NAME} LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(${LIB_NAME}_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(${LIB_NAME}_SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(${LIB_NAME}_TEST_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/test)

option(USE_OPENMP "If available, use OpenMP for parallelization." ON)
if(USE_OPENMP)
  find_package(OpenMP)
endif()

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

find_package(CUDA 8.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(IGRAPH REQUIRED)
find_package(LIBIGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)

include_directories(${${LIB_NAME}_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${IGRAPH_INCLUDES} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

# add_definitions(${PCL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mtune=native -std=c++0x")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

add_subdirectory(${${LIB_NAME}_SOURCE_DIRS}/${LIB_NAME})
add_subdirectory(${${LIB_NAME}_TEST_DIRS})

