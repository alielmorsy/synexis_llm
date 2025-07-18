set(pybind11_PREFER_RELEASE ON)
pybind11_add_module(synexis_python wrapper.cpp)
target_link_libraries(synexis_python PRIVATE syneaxis)

install(TARGETS synexis_python
        LIBRARY DESTINATION synexis_llm
        RUNTIME DESTINATION synexis_llm
        ARCHIVE DESTINATION synexis_llm
)