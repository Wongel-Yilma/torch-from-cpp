cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(load_model)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(load_model load_model.cpp)
target_link_libraries(load_model "${TORCH_LIBRARIES}")
set_property(TARGET load_model PROPERTY CXX_STANDARD 17)
