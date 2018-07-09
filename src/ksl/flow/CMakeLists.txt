cuda_add_library(pd_scene_flow_cuda SHARED pd_scene_flow_cuda.cu)
add_library(pd_scene_flow SHARED pd_scene_flow.cpp)

target_link_libraries(pd_scene_flow_cuda ${CUDA_LIBRARIES})
target_link_libraries(pd_scene_flow ${OpenCV_LIBS} pd_scene_flow_cuda)

