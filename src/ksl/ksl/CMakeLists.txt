add_library(${LIB_NAME} SHARED offline_sf_ksl.cpp online_sf_ksl.cpp)

target_link_libraries(${LIB_NAME} ${IGRAPH_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} clustering flow)

