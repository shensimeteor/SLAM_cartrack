cmake_minimum_required(VERSION 2.8)

project(cartrack)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#cartrack
add_executable(cartrack cartrack.cc)
target_link_libraries(cartrack ${CERES_LIBRARIES})
