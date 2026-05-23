add_library(emu STATIC
    error.cpp
    logger/logger.cpp
    memory/provider.cpp
    memory/shadow_pages.cpp
    memory/heap.cpp
    memory/l1_cache.cpp
    memory/l2_cache.cpp
    memory/l3_cache.cpp
    cache/block_cache.cpp
    ir/insn.cpp
    ir/peephole.cpp
    cpu/cpu.cpp
    cpu/flags.cpp
    cpu/fault.cpp
    decoder/decoder.cpp
    decoder/decoder_opcodes.cpp
    decoder/modrm.cpp
    dispatcher/dispatcher.cpp
    handlers/data_mov.cpp
    handlers/arithmetic.cpp
    handlers/logical.cpp
    handlers/shift_rotate.cpp
    handlers/control_flow.cpp
    handlers/bit_manip.cpp
    handlers/string_ops.cpp
    handlers/sse.cpp
    handlers/sse_fp.cpp
    handlers/x87.cpp
    handlers/avx.cpp
    handlers/misc.cpp
    handlers/sse4_aes.cpp
    handlers/bmi.cpp
    handlers/phase10.cpp
    handlers/avx512.cpp
    handlers/syscalls.cpp
    handlers/gather.cpp
    float80.cpp
    iat_stubs.cpp
)

target_include_directories(emu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Public dependencies: nothing yet. The core stays host-agnostic.
target_link_libraries(emu PUBLIC)
