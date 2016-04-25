# Main Executable
add_executable(lstm-play main.cc)
set_property(TARGET lstm-play PROPERTY CXX_STANDARD 14)

# CNN
add_dependencies(lstm-play cnn)
target_link_libraries(lstm-play ${Boost_LIBRARIES})
target_link_libraries(lstm-play ${CMAKE_BINARY_DIR}/cnn/lib/libcnn.a)
target_include_directories(lstm-play PUBLIC /usr/local/include/eigen3)