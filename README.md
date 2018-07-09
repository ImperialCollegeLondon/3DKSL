# 3DMS
3D Motion Segmentation of Articulated Rigid Bodies based on RGB-D data

# Overview

3DMS performs 3D motion segmentation of articulated rigid bodies from a single-view RGB-D data sequence.

Paper will soon be available.

# Requirements
- Dependencies
1. PCL library: http://pointclouds.org/downloads/linux.html
2. OpenCV: https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html
3. Cuda: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
4. Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page#Download
5. libigl: https://github.com/libigl/libigl
6. Spectra: https://spectralib.org
- General
1. git clone https://github.com/ImperialCollegeLondon/3DMS.git
2. cd 3DMS
3. mkdir build && cd build
4. cmake .. -DCMAKE_BUILD_TYPE=Release && make
- Usage
An executable is provided as an example and it will be located in the bin directory. A path to the directory containing the RGB data sequence must be provided, as well as a path to the directory containing the corresponding Depth data sequence. A last parameter must be provided which corresponds to how many points must be initially sub-sampled from the point cloud (e.g. 1000-1500).
- Example
./example_offline_sf_ksl <path-to-RGB-dir> <path-toDepth-dir> number-sub-samples

This package relies on the following algorithms:
1. Sparse Subspace Clustering: http://vision.jhu.edu/code/
2. Self-Tuning Spectral Clustering: http://www.vision.caltech.edu/lihi/Demos/SelfTuningClustering.html
3. Primal-Dual Scene Flow: https://vision.in.tum.de/research/sceneflow

