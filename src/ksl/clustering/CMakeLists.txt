add_library(clustering SHARED clustering.cpp
	hungarian.cpp
	kmeans.cpp
	label_propagation.cpp
	spectral_clustering.cpp)

target_link_libraries(clustering affinity)

