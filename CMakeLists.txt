project(transposer LANGUAGES CXX)
cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

aux_source_directory(. SRC_LIST)
add_executable(transposer ${SRC_LIST})

target_link_libraries(transposer ${MPI_LIBRARIES})
if(MPI_COMPILE_FLAGS)
  set_target_properties(transposer PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(transposer PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

