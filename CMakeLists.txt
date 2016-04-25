cmake_minimum_required (VERSION 3.2)

project (lstm-play)

include(ExternalProject)
ExternalProject_Add(cnn
  GIT_REPOSITORY    https://github.com/clab/cnn
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_SOURCE_DIR}/ext/cnn"
  CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/cnn/  
)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(src)


