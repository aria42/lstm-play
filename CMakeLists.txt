cmake_minimum_required (VERSION 3.2)

project (lstm-play)

include(ExternalProject)
ExternalProject_Add(cnn
  GIT_REPOSITORY    https://github.com/clab/cnn
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_SOURCE_DIR}/ext/cnn"
  CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/cnn/  
)


# Boost
find_package(Boost 1.55 REQUIRED system serialization program_options)

add_subdirectory(src)


