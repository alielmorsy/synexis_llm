cmake_minimum_required(VERSION 3.13)
project(synexis C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SYNEAXIS_VERSION 0.0.1)

add_subdirectory(vendor/llama.cpp)
add_subdirectory(vendor/pybind11)
add_subdirectory(synexis)
add_subdirectory(python-wrapper)
add_executable(synexis_app main.cpp)
target_link_libraries(synexis_app PRIVATE mtmd)
target_link_libraries(synexis_app PRIVATE syneaxis)

