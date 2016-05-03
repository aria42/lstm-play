# Main Executable
add_executable(lstm-play main.cc)
set_property(TARGET lstm-play PROPERTY CXX_STANDARD 14)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils)

# CNN
add_dependencies(lstm-play cnn utils)
find_package(Boost 1.55 REQUIRED system serialization program_options unit_test_framework container)
target_link_libraries(lstm-play ${Boost_LIBRARIES})
target_link_libraries(lstm-play ${CMAKE_BINARY_DIR}/cnn/lib/libcnn.a)
target_include_directories(lstm-play PUBLIC /usr/local/include/eigen3)
target_include_directories(lstm-play PUBLIC utils/include)
target_include_directories(lstm-play PUBLIC ${CMAKE_BINARY_DIR}/utils/include)