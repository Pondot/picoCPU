// math_target.cpp
//
// Standalone target for emulator benchmarks. Exposes two functions:
//
//   target_ackermann_fn(packed)
//     Win64 ABI: RCX = (m << 32) | n.
//     Returns Ackermann A(m, n) in RAX. With m=3, n=6 -> A(3,6) = 509.
//     Stress-tests CALL/RET, deep recursion (~172,000 nested calls).
//
//   target_zeta_fn(N)
//     Win64 ABI: RCX = N.
//     Returns IEEE-754 bit pattern of sum_{i=1..N} 1/i^2 in RAX.
//     Stress-tests SSE FP (cvtsi2sd, divsd, mulsd, addsd) in a tight loop.
//
// Run, note the printed addresses + expected return values, point your
// emulator at the process. Emulate each function, compare RAX bit-exact.
//
// Build:  cl /O2 /EHsc math_target.cpp /link /OUT:math_target.exe

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <windows.h>

// --- Ackermann ---
#pragma optimize("", off)
extern "C" __declspec(noinline)
uint64_t target_ackermann_core(uint64_t m, uint64_t n) {
    if (m == 0) return n + 1;
    if (n == 0) return target_ackermann_core(m - 1, 1);
    return target_ackermann_core(m - 1, target_ackermann_core(m, n - 1));
}

extern "C" __declspec(noinline)
uint64_t target_ackermann_fn(uint64_t packed) {
    return target_ackermann_core((packed >> 32) & 0xFFFFFFFFu,
                                  packed        & 0xFFFFFFFFu);
}

extern "C" __declspec(noinline)
void target_ackermann_fn_end() { volatile char c = 0; (void)c; }

// --- Riemann Zeta(2) ---
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

int main() {
    auto ackfn   = reinterpret_cast<uintptr_t>(&target_ackermann_fn);
    auto ackend  = reinterpret_cast<uintptr_t>(&target_ackermann_fn_end);
    auto zetfn   = reinterpret_cast<uintptr_t>(&target_zeta_fn);
    auto zetend  = reinterpret_cast<uintptr_t>(&target_zeta_fn_end);

    uint64_t ack_in  = (static_cast<uint64_t>(3) << 32) | static_cast<uint64_t>(6);
    uint64_t ack_exp = target_ackermann_fn(ack_in);     // A(3,6) = 509 = 0x1FD
    uint64_t zet_exp = target_zeta_fn(10000);            // ~1.6448340718

    std::printf("pid=%lu\n", ::GetCurrentProcessId());
    std::printf("ackermann_fn: addr=0x%016llx size=%llu input=0x%016llx expected=0x%016llx\n",
                (unsigned long long)ackfn,
                (unsigned long long)(ackend - ackfn),
                (unsigned long long)ack_in,
                (unsigned long long)ack_exp);
    std::printf("zeta_fn: addr=0x%016llx size=%llu input=N=10000 expected=0x%016llx\n",
                (unsigned long long)zetfn,
                (unsigned long long)(zetend - zetfn),
                (unsigned long long)zet_exp);
    std::fflush(stdout);

    ::Sleep(10 * 60 * 1000);   // park for 10 min so the emulator can RPM us
    return 0;
}
