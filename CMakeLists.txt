cmake_minimum_required(VERSION 3.0)

set(LIB_NAME ksl)
project(${LIB_NAME} LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(KSL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(KSL_SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(KSL_TEST_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/test)

find_package(CUDA 8.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LIBIGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(${KSL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "-Wall -mtune=native -std=c++0x -pthread")
set(CUDA_NVCC_FLAGS "--std=c++11")
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

set(CMAKE_BUILD_TYPE "Release")
message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

add_subdirectory(${KSL_SOURCE_DIRS}/${LIB_NAME})
add_subdirectory(${KSL_TEST_DIRS})

#install(FILES ${HEADERS} DESTINATION include)

