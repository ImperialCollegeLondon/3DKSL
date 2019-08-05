# 3DKSL
Repository for Kinematic Structure Learning (KSL) methods of articulated rigid objects.
For more information and more open-source software please visit the Personal Robotic Lab's website: <https://www.imperial.ac.uk/personal-robotics/software/>.

## Offline 3DKSL
Offline 3DKSL performs motion segmentation of articulated rigid bodies from a batch of RGB-D data sequence.
If you use this code in a scientific publication, please cite the following paper:


```
@inproceedings{nunes18motionseg,
  author    = {Urbano Miguel Nunes and Yiannis Demiris},
  booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
  title     = {3D Motion Segmentation of Articulated Rigid Bodies based on RGB-D Data},
  year      = {2018}
}
```

## Online 3DKSL
Online 3DKSL performs motion segmentation of articulated rigid bodies from RGB-D data in a frame-by-frame basis.
If you use this code in a scientific publication, please cite the following paper:

```
@inproceedings{nunes2019online,
    title           = {Online Unsupervised Learning of the 3D Kinematic Structure of Arbitrary Rigid
Bodies},
    author          = {Urbano Miguel Nunes and Yiannis Demiris},
    booktitle       = {IEEE International Conference on Computer Vision ({ICCV})},
    year            = {2019, to appear},
    organization    = {IEEE}
}
```

# Requirements
- Dependencies
1. PCL library: <http://pointclouds.org/downloads/linux.html>
2. OpenCV: <https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html>
3. Cuda: <http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>
4. Eigen: <http://eigen.tuxfamily.org/index.php?title=Main_Page#Download>
5. libigl: <https://github.com/libigl/libigl>
6. Spectra: <https://spectralib.org>
7. igraph: <https://igraph.org/c/>
- General

On a terminal:
1. git clone <https://github.com/ImperialCollegeLondon/3DKSL.git>
2. cd 3DKSL
3. mkdir build && cd build
4. cmake .. -DCMAKE_BUILD_TYPE=Release && make

# Offline 3DKSL Example
An executable is provided as an example, which will be located in the bin directory.
A path to the directory containing the RGB data sequence must be provided, as well as a path to the directory containing the corresponding Depth data sequence.
A last parameter must be provided which corresponds to how many points must be initially sub-sampled from the point cloud (e.g. 1000-1500).
- Example

./example_offline_sf_ksl path-to-RGB-dir path-toDepth-dir number-sub-samples

# Online 3DKSL Example
An executable is provided as an example, which will be located in the bin directory.
A path to the directory containing the RGB data sequence must be provided, as well as a path to the directory containing the corresponding Depth data sequence.
Optional parameters may be passed, please check test/main_online_sf_ksl.cpp.

- Example

./example_online_sf_ksl path-to-RGB-dir path-toDepth-dir

# License
This project is licensed under the MIT License - see the LICENSE file for details.

