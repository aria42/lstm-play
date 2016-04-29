# Library
add_library(utils SHARED include/utils/io.h)
set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(utils PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/utils>"
        "$<INSTALL_INTERFACE:include>" )

# Boost
find_package(Boost 1.55 REQUIRED system serialization program_options unit_test_framework container)
file(COPY include DESTINATION ${CMAKE_BINARY_DIR}/utils)

# Tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
add_executable(utils-test test/test.cpp)
add_dependencies(utils-test utils)
target_link_libraries(utils-test ${Boost_LIBRARIES})
target_link_libraries(utils-test utils)