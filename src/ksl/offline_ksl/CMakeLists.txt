add_library(offline_sf_ksl SHARED offline_sf_ksl.cpp)

target_link_libraries(offline_sf_ksl ${PCL_LIBRARIES} sparse_subspace_clustering pd_scene_flow)

