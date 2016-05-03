# Library
add_library(utils INTERFACE)

# Boost
find_package(Boost 1.55 REQUIRED system serialization unit_test_framework container)
file(COPY include DESTINATION ${CMAKE_BINARY_DIR}/utils)

# Tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
add_executable(utils-test test/test.cpp)
add_dependencies(utils-test utils)
target_link_libraries(utils-test ${Boost_LIBRARIES})
target_link_libraries(utils-test utils)
target_include_directories(utils-test PUBLIC include/utils)