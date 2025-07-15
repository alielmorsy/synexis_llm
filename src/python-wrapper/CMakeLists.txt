pybind11_add_module(synexis_python wrapper.cpp)
target_link_libraries(synexis_python PRIVATE syneaxis)
set_target_properties(synexis_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)