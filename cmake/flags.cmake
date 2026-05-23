# MSVC-only flags. We target Windows x86-64 exclusively.

if(NOT MSVC)
    message(FATAL_ERROR "This project targets MSVC only. Use Visual Studio 2022 Developer Command Prompt.")
endif()

# Common flags for all configs.
add_compile_options(
    /W4              # high warning level
    /permissive-     # strict standard conformance
    /Zc:__cplusplus  # report real __cplusplus value
    /Zc:preprocessor # conforming preprocessor
    /utf-8           # source + execution charset
    /EHsc            # standard C++ exceptions (cold path only)
    /MP              # parallel compile
)

# Treat select warnings as errors.
add_compile_options(
    /we4715  # not all control paths return a value
    /we4716  # must return a value
    /we4456  # declaration hides previous local
    /we4457  # declaration hides function parameter
    /we4458  # declaration hides class member
)

# Suppress noise.
add_compile_options(
    /wd4324  # structure was padded due to alignment specifier
    /wd4201  # nonstandard nameless struct/union (Win SDK headers)
)

add_compile_definitions(
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

# Per-config tuning.
set(CMAKE_C_FLAGS_DEBUG          "/MDd /Od /Zi /RTC1 /D_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG        "/MDd /Od /Zi /RTC1 /D_DEBUG")
set(CMAKE_C_FLAGS_RELEASE        "/MD /O2 /Oi /Ot /Gy /DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE      "/MD /O2 /Oi /Ot /Gy /DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "/MD /O2 /Oi /Ot /Gy /Zi /DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2 /Oi /Ot /Gy /Zi /DNDEBUG")

# Linker flags.
set(CMAKE_EXE_LINKER_FLAGS_RELEASE        "/INCREMENTAL:NO /OPT:REF /OPT:ICF")
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "/DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")

message(STATUS "Emulator: MSVC ${MSVC_VERSION}, default config ${CMAKE_BUILD_TYPE}")
