// End-to-end test: bake the target_fn byte sequence (the exact code we
// observed in the live target.exe) into a FakeMemoryProvider, run the
// emulator, and check RAX. Independent of whether target.exe is running.

#include "test_framework.h"

#include "emu/cpu.h"
#include "emu/emu.h"
#include "emu/ir.h"
#include "emu/memory.h"

using namespace emu;

namespace {

// The exact 96 bytes of target_fn as observed in the live build.
// We capture them here so the unit tests don't depend on target.exe running.
const std::vector<u8> kTargetFnBytes = {
    0x48, 0x8B, 0xC1,                                                 // mov rax, rcx
    0x48, 0xB9, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5,        // mov rcx, 0xa5a5a5a5a5a5a5a5
    0x48, 0x33, 0xC1,                                                 // xor rax, rcx
    0x48, 0xB9, 0xEB, 0x83, 0xB5, 0x80, 0x46, 0x86, 0xC8, 0x61,        // mov rcx, 0x61c8864680b583eb
    0x48, 0x2B, 0xC1,                                                 // sub rax, rcx
    0x48, 0xB9, 0xB9, 0xE5, 0xE4, 0x1C, 0x6D, 0x47, 0x58, 0xBF,        // mov rcx, 0xbf58476d1ce4e5b9
    0x48, 0xC1, 0xC8, 0x2F,                                            // ror rax, 0x2f
    0x48, 0x0F, 0xAF, 0xC1,                                            // imul rax, rcx
    0x48, 0x8B, 0xC8,                                                 // mov rcx, rax
    0x48, 0xC1, 0xE9, 0x1E,                                            // shr rcx, 0x1e
    0x48, 0x33, 0xC8,                                                 // xor rcx, rax
    0x48, 0xB8, 0xEB, 0x11, 0x31, 0x13, 0xBB, 0x49, 0xD0, 0x94,        // mov rax, 0x94d049bb133111eb
    0x48, 0x0F, 0xAF, 0xC8,                                            // imul rcx, rax
    0x48, 0x8B, 0xC1,                                                 // mov rax, rcx
    0x48, 0xC1, 0xE8, 0x1F,                                            // shr rax, 0x1f
    0x48, 0x33, 0xC1,                                                 // xor rax, rcx
    0xC3,                                                              // ret
    // Padding to round to 96 bytes
    0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
    0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
};

// Reference impl -- must match target_fn() in src/target/target.cpp.
u64 reference_mix(u64 seed) noexcept {
    u64 x = seed;
    x ^= 0xA5A5A5A5A5A5A5A5ull;
    x += 0x9E3779B97F4A7C15ull;
    x  = (x << 17) | (x >> 47);
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= (x >> 30);
    x *= 0x94D049BB133111EBull;
    x ^= (x >> 31);
    return x;
}

u64 run_emu(u64 seed) {
    constexpr GuestAddr kBase = 0x100000;
    FakeMemoryProvider mp;
    auto bytes = kTargetFnBytes;
    if (fail(mp.add_region(kBase, std::move(bytes)))) {
        return 0xBAD0BAD0BAD0BAD0ull;
    }
    Emulator e;
    e.set_memory_read(&mp);
    e.cpu().set_r64(reg::RCX, seed);
    RunResult r = e.run(kBase);
    if (fail(r.status)) {
        std::printf("  emu status=%.*s rip=0x%llx\n",
                    static_cast<int>(to_string(r.status).size()),
                    to_string(r.status).data(),
                    static_cast<unsigned long long>(r.final_rip));
    }
    return r.final_rax;
}

} // namespace

TEST(e2e_mixer_seed_1234) {
    const u64 expected = reference_mix(1234ull);
    EXPECT_HEX(run_emu(1234ull), expected);
}

TEST(e2e_mixer_seed_0) {
    const u64 expected = reference_mix(0ull);
    EXPECT_HEX(run_emu(0ull), expected);
}

TEST(e2e_mixer_seed_minus_1) {
    const u64 seed = static_cast<u64>(-1);
    EXPECT_HEX(run_emu(seed), reference_mix(seed));
}

TEST(e2e_mixer_seed_random_pool) {
    const u64 seeds[] = {
        0x0000000000000001ull,
        0xCAFEBABEDEADBEEFull,
        0x12345678ABCDEF00ull,
        0x7FFFFFFFFFFFFFFFull,
        0x8000000000000000ull,
        0x5555555555555555ull,
        0xAAAAAAAAAAAAAAAAull,
    };
    for (u64 s : seeds) {
        const u64 expected = reference_mix(s);
        const u64 got      = run_emu(s);
        if (got != expected) {
            std::printf("  seed=0x%016llx expected=0x%016llx got=0x%016llx\n",
                        (unsigned long long)s, (unsigned long long)expected,
                        (unsigned long long)got);
            ++__fails;
        }
    }
}
