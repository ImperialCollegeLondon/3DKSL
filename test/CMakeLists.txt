add_executable(example_offline_sf_ksl main_offline_sf_ksl.cpp)
add_executable(example_online_sf_ksl main_online_sf_ksl.cpp)

target_link_libraries(example_offline_sf_ksl ${LIB_NAME})
target_link_libraries(example_online_sf_ksl ${LIB_NAME})

