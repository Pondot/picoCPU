add_executable(tester tester.cpp)

target_link_libraries(tester
    PRIVATE
        emu
        host
)
