cuda_add_library(flow_cuda SHARED pd_scene_flow_cuda.cu)
add_library(flow SHARED pd_scene_flow.cpp)

target_link_libraries(flow_cuda ${CUDA_LIBRARIES})
target_link_libraries(flow ${OpenCV_LIBS} flow_cuda)

