add_executable(tests_unit
    unit/test_main.cpp
    unit/test_decoder.cpp
    unit/test_handlers.cpp
    unit/test_emu_e2e.cpp
    unit/test_flags_crosscheck.cpp
    unit/test_phase9.cpp
    unit/test_phase10.cpp
    unit/test_phase12.cpp
    unit/test_phase13.cpp
    unit/test_fuzz.cpp
)

target_include_directories(tests_unit
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(tests_unit PRIVATE emu)

add_test(NAME tests_unit COMMAND tests_unit)
