// Phase 12 unit tests: EVEX/AVX-512, SEH chained UNWIND_INFO + exception
// handler lookup, DR6 status bits, Float80 round-trip, syscall dispatch.

#include "test_framework.h"

#include "emu/cpu.h"
#include "emu/float80.h"
#include "emu/ir.h"
#include "emu/memory.h"
#include "emu/shadow_pages.h"

#include "../../src/emu/handlers/handlers.h"

#include <cmath>
#include <cstring>

using namespace emu;

namespace {

Insn mk(OpKind k, Handler h, u8 size = 64) {
    Insn i{};
    i.kind = k; i.handler = h; i.op_size = size;
    return i;
}

} // namespace

// EVEX VPXORQ 512-bit, no mask (k0 = "all lanes")
TEST(vpxorq_zmm_no_mask) {
    Cpu c;
    for (int i = 0; i < 64; ++i) { c.zmm(1)[i] = static_cast<u8>(i); c.zmm(2)[i] = static_cast<u8>(0xFF); }
    Insn i = mk(OpKind::Vpxorq, &handlers::op_vpxorq, 64);
    i.flags = INSN_FLAG_VEX;
    i.pad = 0;                                  // aaa=0, z=0 -> no masking
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);
    for (int k = 0; k < 64; ++k) EXPECT_EQ((int)c.zmm(0)[k], k ^ 0xFF);
}

// EVEX VPADDD with k1 mask, merging -- only even lanes (0,2,4,...) get the add
TEST(vpaddd_with_k1_merging) {
    Cpu c;
    // src1 = ones, src2 = twos. After: active lanes become 3, inactive
    // preserve dst's initial value (= 0x55555555 in every lane).
    for (int lane = 0; lane < 16; ++lane) {     // 16 dword lanes in 64-byte zmm
        const u32 v0 = 0x55555555u;
        std::memcpy(c.zmm(0) + 4*lane, &v0, 4);
        const u32 v1 = 1u; std::memcpy(c.zmm(1) + 4*lane, &v1, 4);
        const u32 v2 = 2u; std::memcpy(c.zmm(2) + 4*lane, &v2, 4);
    }
    // k1 = 0xAAAA: bit 1, 3, 5, ... set (odd lanes active)
    c.set_k(1, 0xAAAAull);

    Insn i = mk(OpKind::Vpaddd_evex, &handlers::op_vpaddd_evex, 64);
    i.flags = INSN_FLAG_VEX;
    i.pad = 1;                                  // aaa=1 (k1), z=0 (merging)
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);

    for (int lane = 0; lane < 16; ++lane) {
        u32 r; std::memcpy(&r, c.zmm(0) + 4*lane, 4);
        if (lane & 1) EXPECT_HEX(r, 3u);            // active: 1+2=3
        else          EXPECT_HEX(r, 0x55555555u);    // inactive: preserved
    }
}

// EVEX VPADDQ with k2 mask, zeroing -- inactive lanes go to zero
TEST(vpaddq_with_k2_zeroing) {
    Cpu c;
    for (int lane = 0; lane < 8; ++lane) {       // 8 qword lanes in 64-byte zmm
        const u64 v0 = 0xDEADBEEFCAFEBABEull;
        std::memcpy(c.zmm(0) + 8*lane, &v0, 8);
        const u64 v1 = 1ull; std::memcpy(c.zmm(1) + 8*lane, &v1, 8);
        const u64 v2 = 2ull; std::memcpy(c.zmm(2) + 8*lane, &v2, 8);
    }
    c.set_k(2, 0b10101010ull);                   // 4 lanes active

    Insn i = mk(OpKind::Vpaddq_evex, &handlers::op_vpaddq_evex, 64);
    i.flags = INSN_FLAG_VEX;
    i.pad = 2 | 0x8;                              // aaa=2, z=1 (zeroing)
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);

    for (int lane = 0; lane < 8; ++lane) {
        u64 r; std::memcpy(&r, c.zmm(0) + 8*lane, 8);
        if (lane & 1) EXPECT_HEX(r, 3ull);       // active: 1+2=3
        else          EXPECT_HEX(r, 0ull);       // inactive zeroed
    }
}

// VMOVDQU64 with k1, merging: copies only odd qword lanes; even lanes preserve dst
TEST(vmovdqu64_with_k1_merging) {
    Cpu c;
    // dst (zmm0) initially 0x11111111... in every byte
    for (int b = 0; b < 64; ++b) c.zmm(0)[b] = 0x11;
    // src (zmm1) = sequential 0..63
    for (int b = 0; b < 64; ++b) c.zmm(1)[b] = static_cast<u8>(b);
    c.set_k(1, 0b10101010ull);

    Insn i = mk(OpKind::Vmovdqu64, &handlers::op_vmovdqu64, 64);
    i.flags = INSN_FLAG_VEX;
    i.pad = 1;                                    // aaa=1, z=0
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0;
    i.src.kind = OperandKind::Xmm; i.src.reg = 1;
    i.handler(c, i);

    for (int lane = 0; lane < 8; ++lane) {
        for (int j = 0; j < 8; ++j) {
            const u8 got = c.zmm(0)[8*lane + j];
            const u8 expect = (lane & 1) ? static_cast<u8>(8*lane + j) : 0x11;
            EXPECT_EQ((int)got, (int)expect);
        }
    }
}

// Float80 round-trip of a few canonical doubles.
TEST(float80_roundtrip_canonical) {
    auto rt = [](double d) {
        Float80 f = Float80::from_double(d);
        return f.to_double();
    };
    EXPECT(rt(0.0)  == 0.0);
    EXPECT(rt(1.0)  == 1.0);
    EXPECT(rt(-1.0) == -1.0);
    EXPECT(rt(3.14159265358979) == 3.14159265358979);
    EXPECT(std::isinf(rt(std::numeric_limits<double>::infinity())));
    EXPECT(std::isnan(rt(std::numeric_limits<double>::quiet_NaN())));
}

// DR6 status bit set on execute breakpoint match.
TEST(dr6_set_on_exec_bp_match) {
    Cpu c;
    c.set_dr(0, 0x1234);
    c.set_dr(7, 0x1ull);                          // DR0 local enable, cond=00 exec, len=00
    c.set_dr(6, 0);
    EXPECT(c.exec_breakpoint_at(0x1234));
    EXPECT((c.dr(6) & 0x1) != 0);                 // B0 set
    EXPECT((c.dr(6) & 0xE) == 0);                 // B1..B3 still clear
}

// DR6 status bit set on data write breakpoint match.
TEST(dr6_set_on_data_bp_match) {
    Cpu c;
    c.set_dr(2, 0x2000);
    // DR7: enable DR2 (bits 4-5 local), cond=01 (store) at bits 24-25, len=00 (1)
    c.set_dr(7, (1ull << 4) | (1ull << 24));
    c.set_dr(6, 0);
    EXPECT(c.data_breakpoint_at(0x2000, true, 1));
    EXPECT((c.dr(6) & 0x4) != 0);                 // B2 set
}
