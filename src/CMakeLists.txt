cmake_minimum_required(VERSION 3.13)
project(synexis C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SYNEAXIS_VERSION 0.0.1)

add_subdirectory(vendor/llama.cpp)
add_subdirectory(vendor/pybind11)
add_subdirectory(synexis)
add_subdirectory(python-wrapper)

set(SYNEAXIS_PYPROJECT_BUILD OFF CACHE BOOL "Enable when building via pyproject.toml")

message(STATUS "when generator: ${CMAKE_GENERATOR}")
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")


function(llama_cpp_python_install_target target)
    if (NOT TARGET ${target})
        message("No Target ${target}")
        return()
    endif ()

    install(
            TARGETS ${target}
            LIBRARY DESTINATION "synexis_llm/lib"
            RUNTIME DESTINATION "synexis_llm/lib"
            ARCHIVE DESTINATION "synexis_llm/lib"
    )
    set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
    if (UNIX)
        if (APPLE)
            set_target_properties(${target} PROPERTIES
                    INSTALL_RPATH "@loader_path"
                    BUILD_WITH_INSTALL_RPATH TRUE
            )
        else ()
            set_target_properties(${target} PROPERTIES
                    INSTALL_RPATH "$ORIGIN"
                    BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif ()
    endif ()
endfunction()

if (SYNEAXIS_PYPROJECT_BUILD)
    llama_cpp_python_install_target(llama)
    llama_cpp_python_install_target(ggml)
    llama_cpp_python_install_target(mtmd)

    llama_cpp_python_install_target(ggml-base)

    llama_cpp_python_install_target(ggml-amx)
    llama_cpp_python_install_target(ggml-blas)
    llama_cpp_python_install_target(ggml-can)
    llama_cpp_python_install_target(ggml-cpu)
    llama_cpp_python_install_target(ggml-cuda)
    llama_cpp_python_install_target(ggml-hip)
    llama_cpp_python_install_target(ggml-kompute)
    llama_cpp_python_install_target(ggml-metal)
    llama_cpp_python_install_target(ggml-musa)
    llama_cpp_python_install_target(ggml-rpc)
    llama_cpp_python_install_target(ggml-sycl)
    llama_cpp_python_install_target(ggml-vulkan)

endif ()