set(SYNEAXIS_SOURCES
        sampler/Sampler.cpp
        Synexis.cpp
        SynexisImpl.cpp
        SynexisSlot.cpp
)

add_library(syneaxis STATIC ${SYNEAXIS_SOURCES})

target_link_libraries(syneaxis PRIVATE mtmd)

target_include_directories(syneaxis PUBLIC ../include)