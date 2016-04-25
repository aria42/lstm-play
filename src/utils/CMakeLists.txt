# Library
set(SRC src/io.cc)
set(HDRS include/utils/io.h)
add_library(utils SHARED ${SRC} ${HDRS})

target_include_directories(utils PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/utils>"
        "$<INSTALL_INTERFACE:include>" )

# Boost
find_package(Boost 1.55 REQUIRED system serialization program_options unit_test_framework container)

# Tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
add_executable(utils-test test/test.cpp)
add_dependencies(utils-test utils)
target_link_libraries(utils-test ${Boost_LIBRARIES})
target_link_libraries(utils-test utils)