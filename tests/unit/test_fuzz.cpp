// Decoder fuzz pass -- Phase 7 hardening, deferred until now.
//
// Feed a large quantity of random bytes through `decode_block` and assert
// every result is a valid `Status::*` value (no crash, no SEH, no infinite
// loop). The decoder is supposed to gracefully report an error code on any
// malformed sequence; this test verifies that contract.
//
// The test runs N seeds × M attempts. Each attempt:
//   1. Generates a random byte buffer (up to ~64 bytes long -- bigger than
//      any real instruction can be).
//   2. Calls decode_block on it with a small ProviderBuf backing.
//   3. Asserts the return is one of the known Status values and that the
//      block's byte_size doesn't run past the buffer.
//
// If a particular byte sequence is repeatedly causing a crash, increasing
// the trace seed lets us bisect it. We do not assert any specific behavior
// beyond "no crash, no exit" -- the decoder is allowed to consume bytes,
// fail with UnsupportedOpcode, or report TruncatedInstruction.

#include "test_framework.h"

#include "emu/block_cache.h"
#include "emu/cpu.h"
#include "emu/emu.h"
#include "emu/error.h"
#include "emu/ir.h"
#include "emu/memory.h"
#include "emu/shadow_pages.h"
#include "emu/stub_heap.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

using namespace emu;

namespace {

class FuzzProvider final : public MemoryProvider {
public:
    explicit FuzzProvider(const std::vector<u8>& bytes, GuestAddr base = 0x40000)
        : bytes_(bytes), base_(base) {}

    Status read(GuestAddr addr, usize size, void* out) noexcept override {
        if (addr < base_ || addr + size > base_ + bytes_.size()) {
            return Status::UnmappedRead;
        }
        std::memcpy(out, bytes_.data() + (addr - base_), size);
        return Status::Ok;
    }
    Status write(GuestAddr, usize, const void*) noexcept override {
        return Status::ProtectionViolation;
    }
    bool is_mapped(GuestAddr addr) const noexcept override {
        return addr >= base_ && addr < base_ + bytes_.size();
    }

private:
    std::vector<u8> bytes_;
    GuestAddr       base_;
};

bool is_known_status(Status s) noexcept {
    switch (s) {
        case Status::Ok:
        case Status::InvalidArgument:
        case Status::OutOfMemory:
        case Status::NotImplemented:
        case Status::Internal:
        case Status::UnmappedRead:
        case Status::UnmappedWrite:
        case Status::ProtectionViolation:
        case Status::ProviderFailure:
        case Status::InvalidInstruction:
        case Status::TruncatedInstruction:
        case Status::UnsupportedPrefix:
        case Status::UnsupportedOpcode:
        case Status::HostOpenFailed:
        case Status::HostQueryFailed:
        case Status::HostReadFailed:
        case Status::ProcessNotFound:
        case Status::FunctionNotFound:
            return true;
    }
    return false;
}

} // namespace

TEST(decoder_fuzz_random_bytes) {
    // Try a fixed budget of (seed × attempts × max_size) decodes. With these
    // numbers the test runs in under a second in Release builds.
    constexpr u32 kSeedCount    = 64;
    constexpr u32 kPerSeed      = 16384;
    constexpr u32 kMaxBufBytes  = 96;
    u32 ok_count = 0, fail_count = 0;
    u32 status_counts[20] = {};

    for (u32 seed = 1; seed <= kSeedCount; ++seed) {
        std::mt19937 rng(seed * 0x9E3779B1u);
        for (u32 i = 0; i < kPerSeed; ++i) {
            const u32 size = 1 + (rng() % kMaxBufBytes);
            std::vector<u8> buf(size);
            for (u32 j = 0; j < size; ++j) buf[j] = static_cast<u8>(rng() & 0xFFu);

            FuzzProvider mp(buf);
            DecodedBlock block;
            Status s = decode_block(mp, 0x40000, block, /*max_insns=*/4);

            // Hard contract: return a known Status code, no crash.
            if (!is_known_status(s)) {
                std::printf("FUZZ: unknown status %d for seed=%u i=%u size=%u\n",
                            static_cast<int>(s), seed, i, size);
                EXPECT(false);
                return;
            }
            if (s == Status::Ok) {
                // If decode succeeded, byte_size must not exceed buf.
                if (block.byte_size > size) {
                    std::printf("FUZZ: byte_size=%u exceeds buf=%u (seed=%u i=%u)\n",
                                block.byte_size, size, seed, i);
                    EXPECT(false);
                    return;
                }
                ++ok_count;
            } else {
                ++fail_count;
            }
            const int idx = static_cast<int>(s);
            if (idx >= 0 && idx < 20) ++status_counts[idx];
        }
    }

    const u32 total = kSeedCount * kPerSeed;
    std::printf("fuzz: %u total, %u ok, %u failed (%.1f%% ok)\n",
                total, ok_count, fail_count, 100.0 * ok_count / total);
    EXPECT(ok_count + fail_count == total);
}

// AH/CH/DH/BH high-byte register aliases (no REX prefix, byte operand, reg ∈ 4..7
// means the HIGH byte of AX/CX/DX/BX, not SPL/BPL/SIL/DIL). Regression test for
// the bug that made strlen("Hello!") return 1 -- `test dh, dh` was wrongly testing
// SIL, which was always 0.
TEST(ah_dh_bh_high_byte_aliases) {
    Cpu c;
    // RAX = 0x????_???_??_FF -- set AH explicitly via set_r8(16, ...).
    c.set_r64(reg::RAX, 0x1122334455660000ull);
    c.set_r8(16, 0xAB);                  // set AH
    EXPECT_HEX(c.r(reg::RAX), 0x112233445566AB00ull);
    EXPECT_HEX((u64)c.r8(16), 0xABull);   // read AH

    // CH, DH, BH at indices 17, 18, 19.
    c.set_r64(reg::RCX, 0);
    c.set_r8(17, 0xCD);                  // CH
    EXPECT_HEX(c.r(reg::RCX), 0xCD00ull);
    c.set_r64(reg::RDX, 0xFFFFFFFFFFFFFFFFull);
    c.set_r8(18, 0x00);                  // DH = 0
    EXPECT_HEX(c.r(reg::RDX), 0xFFFFFFFFFFFF00FFull);
    c.set_r64(reg::RBX, 0);
    c.set_r8(19, 0x7F);                  // BH
    EXPECT_HEX(c.r(reg::RBX), 0x7F00ull);

    // Reading AH/DH for the "Hello!" first u64:
    // rdx = 0x0000216F6C6C6548 -> DL = 0x48 ('H'), DH = 0x65 ('e')
    c.set_r64(reg::RDX, 0x0000216F6C6C6548ull);
    EXPECT_HEX((u64)c.r8(reg::RDX), 0x48ull);  // DL via normal r8
    EXPECT_HEX((u64)c.r8(18),       0x65ull);  // DH via high-byte alias
}

TEST(stub_heap_alloc_free_coalesce) {
    StubHeap h;
    h.install(0x10000, 4096);
    EXPECT_HEX(h.free_bytes(), 4096ull);
    EXPECT_HEX(h.used_bytes(), 0ull);

    GuestAddr a = h.alloc(64);   EXPECT(a != 0);
    GuestAddr b = h.alloc(128);  EXPECT(b != 0);
    GuestAddr c = h.alloc(32);   EXPECT(c != 0);
    // Allocations are 16-byte aligned; 32 -> 32, 64 -> 64, 128 -> 128.
    EXPECT_HEX(h.used_bytes(), 64ull + 128ull + 32ull);

    // Free middle, should NOT coalesce (left and right still allocated).
    EXPECT(h.free(b));
    EXPECT(h.block_count() == 4);     // a-used, b-free, c-used, tail-free

    // Free first, coalesces with the freed middle.
    EXPECT(h.free(a));
    // Now: [free 64+128], [c used 32], [tail free]
    EXPECT(h.block_count() == 3);

    // Free last user block, all coalesces with neighbors.
    EXPECT(h.free(c));
    EXPECT(h.block_count() == 1);
    EXPECT_HEX(h.free_bytes(), 4096ull);
    EXPECT_HEX(h.used_bytes(), 0ull);

    // OOM behavior.
    EXPECT(h.alloc(8192) == 0);

    // Double-free returns false.
    GuestAddr d = h.alloc(16);
    EXPECT(h.free(d));
    EXPECT(!h.free(d));
}

// Fault -> recovery: a DIV-by-zero halts the dispatcher; the recovery
// callback gets a chance to consume the fault and redirect RIP to a
// handler. The next iteration runs the handler. Demonstrates the
// emulator's fault-to-SEH plumbing without needing a full PE + .pdata.
TEST(fault_recovery_redirects_rip) {
    // Program at 0x1000:
    //   mov rax, 1          ; B8 01 00 00 00
    //   xor rcx, rcx        ; 48 33 C9
    //   div rcx             ; 48 F7 F1  -> triggers DivideError fault
    //   ret                 ; C3
    // Handler at 0x1100:
    //   mov rax, 0xCAFE     ; B8 FE CA 00 00
    //   ret                 ; C3
    ShadowPages mem(nullptr);
    mem.add_private_region(0x1000, 4096);
    mem.add_private_region(0x3000, 4096);   // stack

    const u8 prog[] = {
        0xB8, 0x01, 0x00, 0x00, 0x00,            // mov eax, 1
        0x48, 0x33, 0xC9,                         // xor rcx, rcx
        0x48, 0xF7, 0xF1,                         // div rcx
        0xC3                                       // ret
    };
    const u8 handler[] = {
        0xB8, 0xFE, 0xCA, 0x00, 0x00,            // mov eax, 0xCAFE
        0xC3                                       // ret
    };
    EXPECT(ok(mem.write(0x1000, sizeof(prog),    prog)));
    EXPECT(ok(mem.write(0x1100, sizeof(handler), handler)));

    Emulator e;
    e.set_memory_read (&mem);
    e.set_memory_write(&mem);

    // Push sentinel return, set RSP.
    constexpr u64 kSentinel = 0xDEAD'BEEF'DEAD'BEEFull;
    u64 rsp = 0x3000 + 4096 - 8;
    u8 buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<u8>((kSentinel >> (8 * i)) & 0xFFu);
    (void)mem.write(rsp, 8, buf);
    e.cpu().set_r64(reg::RSP, rsp);
    e.add_stop_addr(kSentinel);

    // Recovery: on any fault, redirect to 0x1100 and consume.
    e.set_fault_recovery(+[](void*, Emulator& e, const Fault&) noexcept {
        e.cpu().set_rip(0x1100);
        return true;
    }, nullptr);

    RunResult r = e.run(0x1000, 1000);
    EXPECT(ok(r.status));
    EXPECT_HEX(r.final_rax, 0xCAFEull);
}

// Multi-Cpu: drive two independent contexts on shared memory through one
// Emulator. Each thread runs a tiny program independently and sees the
// other's writes. Demonstrates that Emulator::run_with_cpu shares
// block_cache + memory + IatStubs but lets the caller own multiple Cpus.
TEST(multi_cpu_shared_memory) {
    // Program in shared memory at 0x1000:
    //   mov rax, [rcx]         ; load
    //   add rax, 1             ; +1
    //   mov [rcx], rax         ; store back
    //   ret
    // Bytes:  48 8B 01   48 FF C0   48 89 01   C3
    ShadowPages mem(nullptr);
    mem.add_private_region(0x1000, 4096);
    mem.add_private_region(0x2000, 4096);    // shared counter cell
    mem.add_private_region(0x3000, 4096);    // thread A stack
    mem.add_private_region(0x4000, 4096);    // thread B stack
    const u8 prog[] = {0x48, 0x8B, 0x01, 0x48, 0xFF, 0xC0, 0x48, 0x89, 0x01, 0xC3};
    EXPECT(ok(mem.write(0x1000, sizeof(prog), prog)));
    // Initial counter = 0.
    const u64 zero = 0;
    EXPECT(ok(mem.write(0x2000, 8, &zero)));

    Emulator e;
    e.set_memory_read (&mem);
    e.set_memory_write(&mem);

    Cpu thread_a, thread_b;
    thread_a.attach_memory_read(&mem);
    thread_a.attach_memory_write(&mem);
    thread_b.attach_memory_read(&mem);
    thread_b.attach_memory_write(&mem);

    auto run_once = [&](Cpu& cpu, GuestAddr stack_top) noexcept {
        // Push sentinel return, set RCX=counter address.
        constexpr u64 kSentinel = 0xDEADBEEFDEADBEEFull;
        u64 rsp = stack_top - 8;
        u8 buf[8];
        for (int i = 0; i < 8; ++i) buf[i] = static_cast<u8>((kSentinel >> (8 * i)) & 0xFFu);
        (void)mem.write(rsp, 8, buf);
        cpu.set_r64(reg::RSP, rsp);
        cpu.set_r64(reg::RCX, 0x2000);
        e.clear_stop_addrs();
        e.add_stop_addr(kSentinel);
        return e.run_with_cpu(cpu, 0x1000, 1000);
    };

    // Each thread runs the program 3 times in alternating fashion.
    // Counter should end at 6.
    for (int i = 0; i < 3; ++i) {
        EXPECT(ok(run_once(thread_a, 0x4000).status));
        EXPECT(ok(run_once(thread_b, 0x5000).status));
    }
    u64 counter = 0;
    EXPECT(ok(mem.read(0x2000, 8, &counter)));
    EXPECT_HEX(counter, 6ull);

    // The block cache should have one entry (the shared program) -- both
    // Cpus decoded through it.
    EXPECT(e.block_cache().size() == 1);
    EXPECT(e.block_cache().hits() >= 1);
}
