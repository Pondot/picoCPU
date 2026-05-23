// Test target -- a self-contained Win64 exe that exposes one well-known
// function to the emulator. The function is a deterministic 64-bit mixer
// (no syscalls, no globals, no SSE), so it's an easy first emulation gate.
//
// Run target.exe; it prints:
//
//   target: pid=NNNN  fn=0xHHHHHHHHHHHHHHHH  size=KK  expected(seed=1234)=0xRRRRRRRRRRRRRRRR
//
// then waits on stdin until the tester is done. The tester reads the function
// bytes by pid + address via ReadProcessMemory.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <windows.h>
#include <intrin.h>     // __readgsqword

// --- The target function. ---------------------------------------------------
//
// Stable Win64 ABI: u64 arg in RCX, returns in RAX. `noinline` so the linker
// doesn't fold it; `extern "C"` so the symbol name is plain. We mark it as
// having a known internal label end so we can measure its size deterministically
// for the diff harness later -- see `target_fn_end` below.

extern "C" __declspec(noinline)
uint64_t target_fn(uint64_t seed) {
    // A small mixing routine that uses ADD, IMUL, XOR, SHL/SHR, OR -- all
    // covered by the Phase-1 handler set.
    uint64_t x = seed;
    x ^= 0xA5A5A5A5A5A5A5A5ull;
    x += 0x9E3779B97F4A7C15ull;        // golden gamma
    x  = (x << 17) | (x >> 47);        // rotate-left 17 (decoder will see shl+shr+or)
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= (x >> 30);
    x *= 0x94D049BB133111EBull;
    x ^= (x >> 31);
    return x;
}

extern "C" __declspec(noinline)
void target_fn_end() {
    // Sentinel that we use as an upper-bound on `target_fn`'s size. Marked
    // noinline so it doesn't disappear.
    volatile char c = 0; (void)c;
}

// --- TEB-read target. ------------------------------------------------------
//
// `mov rax, qword ptr gs:[0x30]` -- on Win64, TEB.NtTib.Self is at offset 0x30,
// so this returns the TEB base address. Used by the GS:[0x30] passthrough
// gate: tester runs this in the emulator with `cpu.gs_base` set to the
// target's TEB and asserts the result equals that TEB.

extern "C" __declspec(noinline)
uint64_t target_get_teb() {
    return __readgsqword(0x30);
}

extern "C" __declspec(noinline)
void target_get_teb_end() {
    volatile char c = 0; (void)c;
}

// --- Stack-using target. ----------------------------------------------------
//
// `#pragma optimize("", off)` forces MSVC to emit a real stack frame:
//   push rbp; sub rsp, N; mov [rbp-..], reg; ...; add rsp, N; pop rbp; ret
// This exercises the Phase-3 ShadowPages writable stack via PUSH/POP and
// memory writes from the emulator's perspective.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_stack_fn(uint64_t seed) {
    volatile uint64_t a = seed;
    volatile uint64_t b = seed ^ 0xDEADBEEFCAFEBABEull;
    volatile uint64_t c = (a * 31) + b;
    c ^= (a >> 7);
    c += (b << 13);
    return c;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_stack_fn_end() {
    volatile char c = 0; (void)c;
}

// --- Polymorphic target. ---------------------------------------------------
//
// Same math as `target_fn` but expressed with junk instructions interleaved
// -- XCHG with self (assembled as a NOP), MOV reg, reg redundant copies,
// LEA r, [r+0], ADD/SUB inverse pairs -- and with different register
// allocation. The emulator's peephole pass should fold the junk; the
// semantic block hash should make this share a cache slot with the clean
// mixer.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_polymorphic_fn(uint64_t seed) {
    volatile uint64_t x  = seed;       // volatile keeps spurious junk in
    volatile uint64_t j1 = x;           // identity copy
    volatile uint64_t j2 = j1;          // chain of copies
    x = j2;

    x ^= 0xA5A5A5A5A5A5A5A5ull;
    x += 0;                              // identity add
    x += 0x9E3779B97F4A7C15ull;
    x  = (x << 17) | (x >> 47);
    j1 = x; x = j1;                      // redundant
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= (x >> 30);
    x *= 0x94D049BB133111EBull;
    x ^= (x >> 31);
    j2 = x; (void)j2;
    return x;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_polymorphic_fn_end() {
    volatile char c = 0; (void)c;
}

// --- Obfuscated target. ----------------------------------------------------
//
// Uses opaque predicates (CPUID-like checks the compiler can't fold away)
// and a control-flow flattening pattern (state-machine dispatch). The
// emulator's opaque-predicate folder + block cache should still produce the
// right answer -- and ideally cache the hot path.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_obfuscated_fn(uint64_t seed) {
    volatile uint32_t state = 0;
    volatile uint64_t x = seed;

    // Mini state machine. Each case does one mixing step, then sets the next
    // state. State 0xFF means done.
    for (int safety = 0; safety < 64 && state != 0xFF; ++safety) {
        switch (state) {
            case 0:
                x ^= 0xA5A5A5A5A5A5A5A5ull;
                state = 7;             // jump around to confuse decoders
                break;
            case 7:
                x += 0x9E3779B97F4A7C15ull;
                state = 3;
                break;
            case 3:
                x  = (x << 17) | (x >> 47);
                state = 5;
                break;
            case 5:
                x *= 0xBF58476D1CE4E5B9ull;
                state = 9;
                break;
            case 9:
                x ^= (x >> 30);
                state = 11;
                break;
            case 11:
                x *= 0x94D049BB133111EBull;
                state = 13;
                break;
            case 13:
                x ^= (x >> 31);
                state = 0xFF;
                break;
            default:
                state = 0xFF;
                break;
        }
    }
    return x;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_obfuscated_fn_end() {
    volatile char c = 0; (void)c;
}

// --- SSE FP scalar target. -------------------------------------------------
//
// Float math forces MSVC to emit MULSS/ADDSS/CVTSI2SS/etc. Returns the bits
// of the resulting float (cast to u64) so we can compare bit-exact.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_fpscalar_fn(uint64_t seed) {
    volatile float a = static_cast<float>(seed & 0xFFFF);
    volatile float b = 3.14159265f;
    volatile float c = a * b;
    c += static_cast<float>(seed >> 32);
    c = c * c;
    if (c < 0.0f) c = -c;
    uint32_t bits;
    std::memcpy(&bits, const_cast<float*>(&c), sizeof(bits));
    return bits;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_fpscalar_fn_end() { volatile char c = 0; (void)c; }

// --- x87 target. -----------------------------------------------------------
//
// `long double` on MSVC x64 is actually 64-bit double, but using the
// `_control87` intrinsics or pragmas can force the compiler to emit x87
// instructions. The easiest forcing path is the `_FPU_*` macros or
// passing /arch:IA32 -- neither is reliable. We instead use the `_fpclass`
// intrinsic which is implemented via x87 on some configurations.
//
// In practice MSVC x64 prefers SSE for floating point. To deterministically
// exercise our x87 handlers we use a unit-test-only path via the FakeMemory-
// Provider rather than a live target. See tests_unit.exe; no live target.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_x87_fn(uint64_t seed) {
    // This will likely compile to SSE on MSVC x64, so target_x87_fn is also a
    // compile-time double-precision sanity check. The x87 decoder is exercised
    // via hand-crafted byte sequences in unit tests instead.
    volatile double a = static_cast<double>(seed);
    volatile double b = 1.4142135623730951;
    volatile double c = a / b + a * b;
    uint64_t bits;
    std::memcpy(&bits, const_cast<double*>(&c), sizeof(bits));
    return bits;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_x87_fn_end() { volatile char c = 0; (void)c; }

// --- AVX target. -----------------------------------------------------------
//
// Uses AVX intrinsics to exercise VEX-encoded ops in the live binary. MSVC
// emits VPXOR/VPADDQ/VMOVDQU with appropriate /arch:AVX configuration. We
// don't enforce /arch here -- if MSVC falls back to SSE2 the result is still
// correct (just a less interesting exercise).

#include <immintrin.h>

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_avx_fn(uint64_t seed) {
    alignas(32) uint64_t buf[4]  = { seed, seed + 1, seed + 2, seed + 3 };
    alignas(32) uint64_t mask[4] = { 0xA5A5A5A5A5A5A5A5ull, 0x5A5A5A5A5A5A5A5Aull,
                                     0xCAFEBABEDEADBEEFull, 0xDEADBEEFCAFEBABEull };
    alignas(32) uint64_t out[4]  = {};
    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf));
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask));
    __m128i x = _mm_xor_si128(a, b);
    __m128i y = _mm_add_epi64(x, a);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), y);
    return out[0] ^ out[1];
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_avx_fn_end() { volatile char c = 0; (void)c; }

// --- BMI target. -----------------------------------------------------------
//
// Uses BMI intrinsics. MSVC compiles `_andn_u64`, `__bextr_u64`, `_blsr_u64`
// to ANDN/BEXTR/BLSR when /arch:AVX2 (or any BMI1) is set; otherwise falls
// back to library functions. The result is correct either way.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_bmi_fn(uint64_t seed) {
    uint64_t a = seed | 0x100;
    uint64_t b = ~seed | 0x200;
    uint64_t r = (~a) & b;                    // ANDN
    r ^= (a & -static_cast<int64_t>(a));      // BLSI
    r += (a & (a - 1));                       // BLSR
    return r;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_bmi_fn_end() { volatile char c = 0; (void)c; }

// --- Entity-array walker target. -------------------------------------------
//
// Mirrors the screenshot from the original Emulator: target builds an array
// of 18 Entity structs on its heap (HEALTH/ARMOR/TRANSFORM/VELOCITY/AI). A
// walker function strides through the array, accumulating a stable checksum
// of every field. The emulator must read the array via RPM through L1/L2/L3
// and produce the same checksum.
//
// The struct layout is fixed (POD, no padding-dependent fields) and the
// array's base address is printed so the tester can pass it in RCX. Count
// goes in RDX. Walker returns the checksum in RAX.

struct Entity {
    int32_t health;
    int32_t armor;
    int32_t tx, ty, tz;      // transform (int for stability)
    float   vx, vy, vz;      // velocity
    int32_t aggr, aware;     // AI
};

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_entity_walker(const Entity* arr, uint64_t count) {
    // Deterministic per-field mixer. No FP NaN paths so result is bit-stable.
    uint64_t acc = 0xCBF29CE484222325ull;        // FNV-1a offset basis
    for (uint64_t i = 0; i < count; ++i) {
        const Entity& e = arr[i];
        acc ^= static_cast<uint64_t>(static_cast<uint32_t>(e.health));    acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(static_cast<uint32_t>(e.armor));     acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(static_cast<uint32_t>(e.tx));        acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(static_cast<uint32_t>(e.ty));        acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(static_cast<uint32_t>(e.tz));        acc *= 0x100000001B3ull;
        uint32_t vxb, vyb, vzb;
        std::memcpy(&vxb, const_cast<float*>(&e.vx), 4);
        std::memcpy(&vyb, const_cast<float*>(&e.vy), 4);
        std::memcpy(&vzb, const_cast<float*>(&e.vz), 4);
        acc ^= static_cast<uint64_t>(vxb);                                acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(vyb);                                acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(vzb);                                acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(static_cast<uint32_t>(e.aggr));      acc *= 0x100000001B3ull;
        acc ^= static_cast<uint64_t>(static_cast<uint32_t>(e.aware));     acc *= 0x100000001B3ull;
    }
    return acc;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_entity_walker_end() { volatile char c = 0; (void)c; }

// --- Self-modifying-code target. -------------------------------------------
//
// Approach: at startup main() reserves a small RWX buffer that holds a code
// "thunk" template -- `mov eax, ecx; xor eax, imm32; ret`. The target function
// patches the imm32 bytes inside the thunk, then CALLs it. Two patches +
// two calls per invocation; the emulator must invalidate the block cache
// after each write and re-decode the thunk's new bytes.
//
// This avoids any Win32 API calls during the emulated path (no VirtualProtect
// or FlushInstructionCache). The thunk buffer's RWX permission is set up
// once by main() and the emulator just sees writes to a known code range,
// triggers SMC notify, re-decodes.
//
// Layout of the 16-byte thunk:
//   00: 8B C1              mov eax, ecx
//   02: 35 ?? ?? ?? ??     xor eax, imm32   <-- bytes 03..06 patched
//   07: C3                 ret
//   08..15: padding (int 3)

extern "C" {
    // Defined as globals so tester sees them. Set by main() at startup.
    void*    g_selfmod_thunk      = nullptr;
    uint64_t g_selfmod_thunk_size = 16;
}

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_selfmod_fn(uint64_t seed) {
    using ThunkFn = uint32_t (*)(uint32_t);
    uint8_t* buf  = static_cast<uint8_t*>(g_selfmod_thunk);
    ThunkFn  fn   = reinterpret_cast<ThunkFn>(buf);

    // Patch 1: imm32 = 0xAAAAAAAA. Then call.
    buf[3] = 0xAA; buf[4] = 0xAA; buf[5] = 0xAA; buf[6] = 0xAA;
    uint32_t r1 = fn(static_cast<uint32_t>(seed));

    // Patch 2: imm32 = 0x5555AAAA. Then call again.
    buf[3] = 0xAA; buf[4] = 0xAA; buf[5] = 0x55; buf[6] = 0x55;
    uint32_t r2 = fn(static_cast<uint32_t>(seed));

    return (static_cast<uint64_t>(r2) << 32) | static_cast<uint64_t>(r1);
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_selfmod_fn_end() { volatile char c = 0; (void)c; }

// --- SEH target: __try / __except catches a divide-by-zero. -----------------
//
// MSVC compiles SEH on x64 by registering a .pdata UNWIND_INFO entry with
// UNW_FLAG_EHANDLER pointing at the `__C_specific_handler` runtime function;
// the language-specific data is a SCOPE_TABLE describing which RIP ranges
// belong to which __try / __except pair. The emulator's SEH chain walker
// (Phase 9-12) finds the handler RVA from .pdata; the dispatcher's fault
// recovery callback (Phase 16) wires divide-by-zero / int3 / etc. to
// emu_raise_exception(), which sets up the EXCEPTION_RECORD and primes
// __C_specific_handler to run.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_seh_fn(uint64_t seed) {
    uint64_t result = 0;
    __try {
        // seed == 0 -> divide by zero -> handler fires and we land in __except.
        volatile uint64_t divisor = seed;
        volatile uint64_t dividend = 100;
        result = dividend / divisor;
    }
    __except (1) {     // EXCEPTION_EXECUTE_HANDLER
        result = 0xDEADBEEF;
    }
    return result;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_seh_fn_end() { volatile char c = 0; (void)c; }

// --- Ackermann (deep-recursion test). ---------------------------------------
// Classic mathematical "this is what unbounded recursion looks like" target.
// A(0, n) = n + 1
// A(m, 0) = A(m-1, 1)
// A(m, n) = A(m-1, A(m, n-1))
// Stress-tests CALL / RET / stack-frame setup at significant depth.
// Caller packs (m << 32) | n into RCX; we unpack and recurse.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_ackermann_core(uint64_t m, uint64_t n) {
    if (m == 0) return n + 1;
    if (n == 0) return target_ackermann_core(m - 1, 1);
    return target_ackermann_core(m - 1, target_ackermann_core(m, n - 1));
}

extern "C" __declspec(noinline)
uint64_t target_ackermann_fn(uint64_t packed) {
    uint64_t m = (packed >> 32) & 0xFFFFFFFFu;
    uint64_t n =  packed        & 0xFFFFFFFFu;
    return target_ackermann_core(m, n);
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_ackermann_fn_end() { volatile char c = 0; (void)c; }

// --- Riemann Zeta (FP / SIMD test). -----------------------------------------
// ζ(s) = Σ_{n=1..N} 1/n^s
// We hardcode s=2 (so ζ(2) = π²/6 ≈ 1.6449) and let the caller pick N via
// RCX. Result returned as raw double-bits in RAX so the diff harness can
// compare bit-exact. Stress-tests SSE2 scalar FP (mulsd, divsd, cvtsi2sd,
// addsd) plus a tight loop.

#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_zeta_fn(uint64_t N) {
    if (N == 0) N = 1;
    double sum = 0.0;
    for (uint64_t i = 1; i <= N; ++i) {
        double d = static_cast<double>(i);
        sum += 1.0 / (d * d);
    }
    uint64_t bits;
    std::memcpy(&bits, &sum, sizeof(bits));
    return bits;
}
#pragma optimize("", on)

extern "C" __declspec(noinline)
void target_zeta_fn_end() { volatile char c = 0; (void)c; }

int wmain(int /*argc*/, wchar_t** /*argv*/) {
    SetConsoleOutputCP(CP_UTF8);

    auto fn   = reinterpret_cast<uintptr_t>(&target_fn);
    auto fend = reinterpret_cast<uintptr_t>(&target_fn_end);
    auto size = (fend > fn) ? (fend - fn) : 0;

    auto tfn   = reinterpret_cast<uintptr_t>(&target_get_teb);
    auto tend  = reinterpret_cast<uintptr_t>(&target_get_teb_end);
    auto tsize = (tend > tfn) ? (tend - tfn) : 0;

    auto sfn   = reinterpret_cast<uintptr_t>(&target_stack_fn);
    auto send  = reinterpret_cast<uintptr_t>(&target_stack_fn_end);
    auto ssize = (send > sfn) ? (send - sfn) : 0;

    uint64_t expected = target_fn(1234ull);
    uint64_t teb      = target_get_teb();
    uint64_t s_exp    = target_stack_fn(1234ull);

    std::printf("target: pid=%lu  fn=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                ::GetCurrentProcessId(),
                static_cast<unsigned long long>(fn),
                static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(expected));
    std::printf("teb_fn: addr=0x%016llx  size=%llu  teb=0x%016llx  tid=%lu\n",
                static_cast<unsigned long long>(tfn),
                static_cast<unsigned long long>(tsize),
                static_cast<unsigned long long>(teb),
                ::GetCurrentThreadId());
    std::printf("stack_fn: addr=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(sfn),
                static_cast<unsigned long long>(ssize),
                static_cast<unsigned long long>(s_exp));

    auto pfn  = reinterpret_cast<uintptr_t>(&target_polymorphic_fn);
    auto pend = reinterpret_cast<uintptr_t>(&target_polymorphic_fn_end);
    auto psize = (pend > pfn) ? (pend - pfn) : 0;
    uint64_t p_exp = target_polymorphic_fn(1234ull);
    std::printf("poly_fn: addr=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(pfn),
                static_cast<unsigned long long>(psize),
                static_cast<unsigned long long>(p_exp));

    auto ofn  = reinterpret_cast<uintptr_t>(&target_obfuscated_fn);
    auto oend = reinterpret_cast<uintptr_t>(&target_obfuscated_fn_end);
    auto osize = (oend > ofn) ? (oend - ofn) : 0;
    uint64_t o_exp = target_obfuscated_fn(1234ull);
    std::printf("obf_fn: addr=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(ofn),
                static_cast<unsigned long long>(osize),
                static_cast<unsigned long long>(o_exp));

    auto fpfn  = reinterpret_cast<uintptr_t>(&target_fpscalar_fn);
    auto fpend = reinterpret_cast<uintptr_t>(&target_fpscalar_fn_end);
    auto fpsize = (fpend > fpfn) ? (fpend - fpfn) : 0;
    uint64_t fp_exp = target_fpscalar_fn(1234ull);
    std::printf("fp_fn: addr=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(fpfn),
                static_cast<unsigned long long>(fpsize),
                static_cast<unsigned long long>(fp_exp));

    auto x87fn  = reinterpret_cast<uintptr_t>(&target_x87_fn);
    auto x87end = reinterpret_cast<uintptr_t>(&target_x87_fn_end);
    auto x87size = (x87end > x87fn) ? (x87end - x87fn) : 0;
    uint64_t x87_exp = target_x87_fn(1234ull);
    std::printf("x87_fn: addr=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(x87fn),
                static_cast<unsigned long long>(x87size),
                static_cast<unsigned long long>(x87_exp));

    auto avxfn  = reinterpret_cast<uintptr_t>(&target_avx_fn);
    auto avxend = reinterpret_cast<uintptr_t>(&target_avx_fn_end);
    auto avxsize = (avxend > avxfn) ? (avxend - avxfn) : 0;
    uint64_t avx_exp = target_avx_fn(1234ull);
    std::printf("avx_fn: addr=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(avxfn),
                static_cast<unsigned long long>(avxsize),
                static_cast<unsigned long long>(avx_exp));

    auto bmifn  = reinterpret_cast<uintptr_t>(&target_bmi_fn);
    auto bmiend = reinterpret_cast<uintptr_t>(&target_bmi_fn_end);
    auto bmisize = (bmiend > bmifn) ? (bmiend - bmifn) : 0;
    uint64_t bmi_exp = target_bmi_fn(1234ull);
    std::printf("bmi_fn: addr=0x%016llx  size=%llu  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(bmifn),
                static_cast<unsigned long long>(bmiend > bmifn ? bmiend - bmifn : 0),
                static_cast<unsigned long long>(bmi_exp));

    // --- Entity-array gate. Build 18 entities, print the array address and
    // count, run the walker once natively to record the expected checksum.
    static constexpr uint64_t kEntityCount = 18;
    Entity* entities = static_cast<Entity*>(::HeapAlloc(::GetProcessHeap(), 0,
                                                       sizeof(Entity) * kEntityCount));
    // Deterministic seeded fill (no rand()) so the gate is reproducible.
    {
        uint64_t s = 0xC0FFEE00FEEDF00Dull;
        auto next = [&]() noexcept {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            return s;
        };
        for (uint64_t i = 0; i < kEntityCount; ++i) {
            entities[i].health = 30 + static_cast<int32_t>(next() % 130);
            entities[i].armor  = 25;
            entities[i].tx     = static_cast<int32_t>(next() % 257) - 128;
            entities[i].ty     = static_cast<int32_t>(next() % 257) - 128;
            entities[i].tz     = static_cast<int32_t>(next() % 257) - 128;
            entities[i].vx     = (static_cast<float>(next() % 200) - 100.f) / 100.f;
            entities[i].vy     = (static_cast<float>(next() % 200) - 100.f) / 100.f;
            entities[i].vz     = (static_cast<float>(next() % 200) - 100.f) / 100.f;
            entities[i].aggr   = 8 + static_cast<int32_t>(next() % 8);
            entities[i].aware  = 37 + static_cast<int32_t>(next() % 16);
        }
    }
    auto entfn   = reinterpret_cast<uintptr_t>(&target_entity_walker);
    auto entend  = reinterpret_cast<uintptr_t>(&target_entity_walker_end);
    auto entsize = (entend > entfn) ? (entend - entfn) : 0;
    uint64_t ent_exp = target_entity_walker(entities, kEntityCount);
    std::printf("entity_fn: addr=0x%016llx  size=%llu  arr=0x%016llx  count=%llu  expected=0x%016llx\n",
                static_cast<unsigned long long>(entfn),
                static_cast<unsigned long long>(entsize),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(entities)),
                static_cast<unsigned long long>(kEntityCount),
                static_cast<unsigned long long>(ent_exp));

    // --- Self-modifying gate. Allocate an RWX thunk page once. The thunk
    // template is written by main(); target_selfmod_fn patches the imm32
    // and CALLs into it.
    g_selfmod_thunk = ::VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE,
                                     PAGE_EXECUTE_READWRITE);
    uint8_t* tbuf = static_cast<uint8_t*>(g_selfmod_thunk);
    tbuf[0] = 0x8B; tbuf[1] = 0xC1;                          // mov eax, ecx
    tbuf[2] = 0x35; tbuf[3] = 0x00; tbuf[4] = 0x00;
    tbuf[5] = 0x00; tbuf[6] = 0x00;                          // xor eax, imm32 (placeholder)
    tbuf[7] = 0xC3;                                          // ret
    for (int i = 8; i < 16; ++i) tbuf[i] = 0xCC;             // int 3 padding

    auto smfn   = reinterpret_cast<uintptr_t>(&target_selfmod_fn);
    auto smend  = reinterpret_cast<uintptr_t>(&target_selfmod_fn_end);
    auto smsize = (smend > smfn) ? (smend - smfn) : 0;
    uint64_t sm_exp = target_selfmod_fn(1234ull);
    std::printf("selfmod_fn: addr=0x%016llx  size=%llu  thunk=0x%016llx  expected(seed=1234)=0x%016llx\n",
                static_cast<unsigned long long>(smfn),
                static_cast<unsigned long long>(smsize),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_selfmod_thunk)),
                static_cast<unsigned long long>(sm_exp));

    // SEH target -- runs the function twice (seed=10 takes the try path,
    // seed=0 triggers divide-by-zero into the __except path) so the gate
    // tester can pick either case.
    auto sehfn   = reinterpret_cast<uintptr_t>(&target_seh_fn);
    auto sehend  = reinterpret_cast<uintptr_t>(&target_seh_fn_end);
    auto sehsize = (sehend > sehfn) ? (sehend - sehfn) : 0;
    uint64_t seh_ok  = target_seh_fn(10);     // 100/10 = 10
    uint64_t seh_div = target_seh_fn(0);      // catches -> 0xDEADBEEF
    std::printf("seh_fn: addr=0x%016llx  size=%llu  ok(seed=10)=0x%016llx  div(seed=0)=0x%016llx\n",
                static_cast<unsigned long long>(sehfn),
                static_cast<unsigned long long>(sehsize),
                static_cast<unsigned long long>(seh_ok),
                static_cast<unsigned long long>(seh_div));

    // Ackermann gate. A(3, 6) = 509 -- deep recursion (~172,000 calls), still
    // fits comfortably in our 64 KiB emulated stack.
    auto ackfn    = reinterpret_cast<uintptr_t>(&target_ackermann_fn);
    auto ackend   = reinterpret_cast<uintptr_t>(&target_ackermann_fn_end);
    auto acksize  = (ackend > ackfn) ? (ackend - ackfn) : 0;
    uint64_t ack_in = ((uint64_t)3 << 32) | (uint64_t)6;
    uint64_t ack_exp = target_ackermann_fn(ack_in);
    std::printf("ackermann_fn: addr=0x%016llx  size=%llu  A(3,6) input=0x%016llx  expected=0x%016llx\n",
                static_cast<unsigned long long>(ackfn),
                static_cast<unsigned long long>(acksize),
                static_cast<unsigned long long>(ack_in),
                static_cast<unsigned long long>(ack_exp));

    // Zeta gate. ζ(2) ≈ 1.6449 with N = 10,000 terms. Returns the double's
    // bit pattern so the diff harness can compare exactly.
    auto zetfn   = reinterpret_cast<uintptr_t>(&target_zeta_fn);
    auto zetend  = reinterpret_cast<uintptr_t>(&target_zeta_fn_end);
    auto zetsize = (zetend > zetfn) ? (zetend - zetfn) : 0;
    uint64_t zet_exp = target_zeta_fn(10000);
    std::printf("zeta_fn: addr=0x%016llx  size=%llu  zeta(2,N=10000) bits=0x%016llx\n",
                static_cast<unsigned long long>(zetfn),
                static_cast<unsigned long long>(zetsize),
                static_cast<unsigned long long>(zet_exp));
    std::fflush(stdout);

    // Park: always Sleep so we work in backgrounded/redirected launches.
    // Kill the process externally (or wait up to 10 min) when done.
    ::Sleep(10 * 60 * 1000);
    ::HeapFree(::GetProcessHeap(), 0, entities);
    return 0;
}
