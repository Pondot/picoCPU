add_library(host STATIC
    win_process.cpp
    win_tls.cpp
    pe_image.cpp
    pe_loader.cpp
    seh.cpp
)

target_include_directories(host
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(host
    PUBLIC  emu
    PRIVATE psapi
)
