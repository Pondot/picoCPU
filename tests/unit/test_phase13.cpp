// Phase 13 unit tests: AVX-512 FP, EVEX broadcast, AVX2 VGATHER.
//
// Float80 soft-float math tests live alongside.

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

// ---- AVX-512 floating-point ------------------------------------------------

TEST(vaddps_evex_no_mask_512) {
    Cpu c;
    for (int lane = 0; lane < 16; ++lane) {
        const float a = static_cast<float>(lane);
        const float b = static_cast<float>(lane * 2);
        std::memcpy(c.zmm(1) + 4*lane, &a, 4);
        std::memcpy(c.zmm(2) + 4*lane, &b, 4);
    }
    Insn i = mk(OpKind::Vaddps_evex, &handlers::op_vaddps_evex, 64);
    i.flags = INSN_FLAG_VEX; i.pad = 0;
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);
    for (int lane = 0; lane < 16; ++lane) {
        float got; std::memcpy(&got, c.zmm(0) + 4*lane, 4);
        EXPECT(got == static_cast<float>(lane * 3));
    }
}

TEST(vmulpd_evex_with_k1_merging) {
    Cpu c;
    for (int lane = 0; lane < 8; ++lane) {
        const double init = 999.0;
        std::memcpy(c.zmm(0) + 8*lane, &init, 8);
        const double a = static_cast<double>(lane + 1);
        std::memcpy(c.zmm(1) + 8*lane, &a, 8);
        const double b = 2.0;
        std::memcpy(c.zmm(2) + 8*lane, &b, 8);
    }
    c.set_k(1, 0b10101010ull);

    Insn i = mk(OpKind::Vmulpd_evex, &handlers::op_vmulpd_evex, 64);
    i.flags = INSN_FLAG_VEX; i.pad = 1;
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);

    for (int lane = 0; lane < 8; ++lane) {
        double got; std::memcpy(&got, c.zmm(0) + 8*lane, 8);
        const double want = (lane & 1) ? static_cast<double>(2 * (lane + 1)) : 999.0;
        EXPECT(got == want);
    }
}

TEST(vsqrtpd_evex_unop) {
    Cpu c;
    for (int lane = 0; lane < 8; ++lane) {
        const double v = static_cast<double>(lane * lane * 4);
        std::memcpy(c.zmm(2) + 8*lane, &v, 8);
    }
    Insn i = mk(OpKind::Vsqrtpd_evex, &handlers::op_vsqrtpd_evex, 64);
    i.flags = INSN_FLAG_VEX; i.pad = 0;
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);

    for (int lane = 0; lane < 8; ++lane) {
        double got; std::memcpy(&got, c.zmm(0) + 8*lane, 8);
        EXPECT(got == static_cast<double>(lane) * 2.0);
    }
}

TEST(vminps_evex_propagates_second_on_nan) {
    Cpu c;
    const float nan_v = std::nanf("");
    const float val   = 7.0f;
    std::memcpy(c.zmm(1) + 0, &nan_v, 4);     // lane 0 of src1 is NaN
    std::memcpy(c.zmm(2) + 0, &val,   4);     // lane 0 of src2 is 7.0
    std::memcpy(c.zmm(1) + 4, &val,   4);     // lane 1 of src1 is 7.0
    const float small = 3.0f;
    std::memcpy(c.zmm(2) + 4, &small, 4);     // lane 1 of src2 is 3.0
    // Remaining lanes: leave zero / zero so min returns src2 (zero).

    Insn i = mk(OpKind::Vminps_evex, &handlers::op_vminps_evex, 64);
    i.flags = INSN_FLAG_VEX; i.pad = 0;
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);

    float got0, got1;
    std::memcpy(&got0, c.zmm(0) + 0, 4);
    std::memcpy(&got1, c.zmm(0) + 4, 4);
    EXPECT(got0 == 7.0f);     // NaN in src1 -> returns src2
    EXPECT(got1 == 3.0f);     // ordinary min
}

// ---- AVX-512 broadcast -----------------------------------------------------

TEST(vaddpd_evex_broadcast_from_mem) {
    Cpu c;
    ShadowPages pages(nullptr);
    c.attach_memory_read(&pages);
    c.attach_memory_write(&pages);

    // src2 in memory: single 8-byte double = 10.0, broadcast to all 8 lanes.
    const GuestAddr addr = 0x1000;
    const double bcast_val = 10.0;
    u8 bcast_buf[8];
    std::memcpy(bcast_buf, &bcast_val, 8);
    EXPECT(ok(pages.write(addr, 8, bcast_buf)));

    // src1: lane i = i.
    for (int lane = 0; lane < 8; ++lane) {
        const double a = static_cast<double>(lane);
        std::memcpy(c.zmm(1) + 8*lane, &a, 8);
    }

    Insn i = mk(OpKind::Vaddpd_evex, &handlers::op_vaddpd_evex, 64);
    i.flags = INSN_FLAG_VEX;
    i.pad = 0x10;                          // broadcast bit set, no mask
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Mem; i.src.reg = reg::NONE; i.src.index = reg::NONE;
    i.src.imm = static_cast<i64>(addr);
    i.handler(c, i);

    for (int lane = 0; lane < 8; ++lane) {
        double got; std::memcpy(&got, c.zmm(0) + 8*lane, 8);
        EXPECT(got == static_cast<double>(lane) + 10.0);
    }
}

// ---- AVX2 VGATHER ----------------------------------------------------------

TEST(vpgatherdd_xmm_4lanes) {
    Cpu c;
    ShadowPages pages(nullptr);
    c.attach_memory_read(&pages);
    c.attach_memory_write(&pages);

    // Memory layout: base = 0x2000; bytes hold values 0..0xFF at consecutive
    // u32 boundaries.
    const GuestAddr base = 0x2000;
    for (u32 v = 0; v < 64; ++v) {
        u8 buf[4];
        const u32 word = 0xCAFE0000u + v;
        std::memcpy(buf, &word, 4);
        EXPECT(ok(pages.write(base + v * 4u, 4, buf)));
    }

    // index xmm[3] holds 4 dword indices: 1, 3, 5, 7.
    const u32 idx_vals[4] = {1, 3, 5, 7};
    for (int lane = 0; lane < 4; ++lane) {
        std::memcpy(c.ymm(3) + 4*lane, &idx_vals[lane], 4);
    }
    // mask xmm[4]: all 4 lanes active (sign bit set in each dword).
    for (int lane = 0; lane < 4; ++lane) {
        const u32 m = 0x80000000u;
        std::memcpy(c.ymm(4) + 4*lane, &m, 4);
    }
    // base GPR R10 = 0x2000.
    c.set_r64(reg::R10, base);

    Insn i = mk(OpKind::Vpgatherdd, &handlers::op_vpgatherdd, 16);
    i.flags = INSN_FLAG_VEX;
    i.imm_extra = static_cast<i64>(4u | (4u << 8));      // data=4, idx=4
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 4;
    i.src.kind = OperandKind::Mem;
    i.src.reg = reg::R10; i.src.index = 3; i.src.scale = 4; i.src.imm = 0;
    i.src.pad[0] = 1;
    i.handler(c, i);

    // Expected results: lane 0 = mem[base + 1*4] = 0xCAFE0001, etc.
    for (int lane = 0; lane < 4; ++lane) {
        u32 got; std::memcpy(&got, c.ymm(0) + 4*lane, 4);
        EXPECT_HEX(got, 0xCAFE0000u + idx_vals[lane]);
    }
    // Mask should be cleared in all 4 lanes after successful gather.
    for (int lane = 0; lane < 4; ++lane) {
        u32 m; std::memcpy(&m, c.ymm(4) + 4*lane, 4);
        EXPECT_HEX(m, 0u);
    }
}

TEST(vpgatherqq_only_active_lanes_gathered) {
    Cpu c;
    ShadowPages pages(nullptr);
    c.attach_memory_read(&pages);
    c.attach_memory_write(&pages);

    const GuestAddr base = 0x3000;
    for (u64 v = 0; v < 32; ++v) {
        u8 buf[8];
        const u64 word = 0xDEADBEEF00000000ull + v;
        std::memcpy(buf, &word, 8);
        EXPECT(ok(pages.write(base + v * 8u, 8, buf)));
    }
    // 4 qword indices: 2, 4, 6, 8 -- but only lanes 0 and 2 are active.
    const u64 idx_vals[4] = {2, 4, 6, 8};
    for (int lane = 0; lane < 4; ++lane) {
        std::memcpy(c.ymm(5) + 8*lane, &idx_vals[lane], 8);
    }
    // Mask: lanes 0 and 2 active (high bit set), lanes 1 and 3 inactive.
    u64 active = 0x8000000000000000ull;
    u64 inactive = 0;
    std::memcpy(c.ymm(6) + 0,  &active,   8);
    std::memcpy(c.ymm(6) + 8,  &inactive, 8);
    std::memcpy(c.ymm(6) + 16, &active,   8);
    std::memcpy(c.ymm(6) + 24, &inactive, 8);

    // Initialize dst with a sentinel so we can verify inactive lanes are preserved.
    const u64 sentinel = 0x1111111111111111ull;
    for (int lane = 0; lane < 4; ++lane) std::memcpy(c.ymm(0) + 8*lane, &sentinel, 8);

    c.set_r64(reg::R10, base);

    Insn i = mk(OpKind::Vpgatherqq, &handlers::op_vpgatherqq, 32);   // ymm width
    i.flags = INSN_FLAG_VEX;
    i.imm_extra = static_cast<i64>(8u | (8u << 8));
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 6;
    i.src.kind = OperandKind::Mem;
    i.src.reg = reg::R10; i.src.index = 5; i.src.scale = 8; i.src.imm = 0;
    i.src.pad[0] = 1;
    i.handler(c, i);

    u64 got0, got1, got2, got3;
    std::memcpy(&got0, c.ymm(0) + 0,  8);
    std::memcpy(&got1, c.ymm(0) + 8,  8);
    std::memcpy(&got2, c.ymm(0) + 16, 8);
    std::memcpy(&got3, c.ymm(0) + 24, 8);
    EXPECT_HEX(got0, 0xDEADBEEF00000000ull + 2);   // active: gathered
    EXPECT_HEX(got1, sentinel);                    // inactive: preserved
    EXPECT_HEX(got2, 0xDEADBEEF00000000ull + 6);   // active: gathered
    EXPECT_HEX(got3, sentinel);                    // inactive: preserved

    // Active lanes' mask bits cleared, inactive ones preserved.
    u64 m0, m1, m2, m3;
    std::memcpy(&m0, c.ymm(6) + 0,  8);
    std::memcpy(&m1, c.ymm(6) + 8,  8);
    std::memcpy(&m2, c.ymm(6) + 16, 8);
    std::memcpy(&m3, c.ymm(6) + 24, 8);
    EXPECT_HEX(m0, 0ull);                          // cleared
    EXPECT_HEX(m1, 0ull);                          // already zero
    EXPECT_HEX(m2, 0ull);                          // cleared
    EXPECT_HEX(m3, 0ull);                          // already zero
}

// ---- 80-bit soft-float math ------------------------------------------------

TEST(f80_add_basic) {
    auto a = Float80::from_double(1.5);
    auto b = Float80::from_double(2.25);
    auto r = f80_add(a, b);
    EXPECT(r.to_double() == 3.75);
}

TEST(f80_add_different_signs_cancels) {
    auto a = Float80::from_double(5.0);
    auto b = Float80::from_double(-5.0);
    auto r = f80_add(a, b);
    EXPECT(r.to_double() == 0.0);
}

TEST(f80_add_inf_minus_inf_is_nan) {
    auto inf_pos = Float80::inf(false);
    auto inf_neg = Float80::inf(true);
    auto r = f80_add(inf_pos, inf_neg);
    EXPECT(r.is_nan());
}

TEST(f80_sub_basic) {
    auto a = Float80::from_double(10.0);
    auto b = Float80::from_double(3.5);
    auto r = f80_sub(a, b);
    EXPECT(r.to_double() == 6.5);
}

TEST(f80_sub_borrows_renormalizes) {
    // 1.0 - 0.5 = 0.5 -- exercises shift-left renormalization.
    auto a = Float80::from_double(1.0);
    auto b = Float80::from_double(0.5);
    auto r = f80_sub(a, b);
    EXPECT(r.to_double() == 0.5);
}

TEST(f80_mul_basic) {
    auto a = Float80::from_double(3.0);
    auto b = Float80::from_double(4.0);
    auto r = f80_mul(a, b);
    EXPECT(r.to_double() == 12.0);
}

TEST(f80_mul_signed) {
    auto a = Float80::from_double(-2.5);
    auto b = Float80::from_double(4.0);
    auto r = f80_mul(a, b);
    EXPECT(r.to_double() == -10.0);
}

TEST(f80_mul_zero_inf_is_nan) {
    auto z   = Float80::zero(false);
    auto inf = Float80::inf(false);
    auto r   = f80_mul(z, inf);
    EXPECT(r.is_nan());
}

TEST(f80_div_basic) {
    auto a = Float80::from_double(20.0);
    auto b = Float80::from_double(4.0);
    auto r = f80_div(a, b);
    EXPECT(r.to_double() == 5.0);
}

TEST(f80_div_by_zero_is_inf) {
    auto a = Float80::from_double(7.0);
    auto z = Float80::zero(false);
    auto r = f80_div(a, z);
    EXPECT(r.is_inf());
    EXPECT(!r.is_sign_negative());
}

TEST(f80_div_zero_by_zero_is_nan) {
    auto z = Float80::zero(false);
    auto r = f80_div(z, z);
    EXPECT(r.is_nan());
}

// AVX-512 VPSCATTERDD with k1 mask -- store qwords from src to indexed lanes.
TEST(vpscatterdd_xmm_4_lanes) {
    Cpu c;
    ShadowPages pages(nullptr);
    c.attach_memory_read(&pages);
    c.attach_memory_write(&pages);
    pages.add_private_region(0x4000, 4096);

    // dst (data source): zmm[5] has 4 dwords: 0xAAAA0001, 0xAAAA0002, ...
    for (int lane = 0; lane < 4; ++lane) {
        const u32 v = 0xAAAA0001u + static_cast<u32>(lane);
        std::memcpy(c.zmm(5) + 4 * lane, &v, 4);
    }
    // index reg: zmm[7] has 4 dwords: 1, 3, 5, 7.
    for (int lane = 0; lane < 4; ++lane) {
        const u32 idx = 1u + static_cast<u32>(lane) * 2u;
        std::memcpy(c.zmm(7) + 4 * lane, &idx, 4);
    }
    // base GPR R10 = 0x4000.
    c.set_r64(reg::R10, 0x4000);
    c.set_k(1, 0b1111ull);    // all 4 lanes active

    Insn i{};
    i.kind     = OpKind::Vpscatterdd;
    i.handler  = &handlers::op_vpscatterdd;
    i.op_size  = 16;          // xmm width
    i.flags    = INSN_FLAG_VEX;
    i.pad      = 1;           // aaa=1, no zeroing, no broadcast
    i.imm_extra = static_cast<i64>(4u | (4u << 8));
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 5;        // source data
    i.src.kind = OperandKind::Mem;
    i.src.reg = reg::R10; i.src.index = 7; i.src.scale = 4; i.src.imm = 0;
    i.src.pad[0] = 1;
    i.handler(c, i);

    // Lane i should write 0xAAAA000{i+1} to address 0x4000 + (1+2i)*4.
    for (int lane = 0; lane < 4; ++lane) {
        u8 buf[4];
        const GuestAddr addr = 0x4000 + (1u + static_cast<u32>(lane) * 2u) * 4u;
        EXPECT(ok(pages.read(addr, 4, buf)));
        u32 got = 0;
        for (int j = 0; j < 4; ++j) got |= (u32(buf[j]) << (8 * j));
        EXPECT_HEX(got, 0xAAAA0001u + static_cast<u32>(lane));
    }
    // Mask should be cleared after successful stores.
    EXPECT_HEX(c.k(1), 0ull);
}

// 80-bit transcendentals -- sin/cos/2^x/log2/scale at extended precision.

TEST(f80_sin_basic_angles) {
    auto sin0 = f80_sin(Float80::from_double(0.0));
    EXPECT(sin0.to_double() == 0.0);
    auto sin_pi = f80_sin(Float80::from_double(3.14159265358979323846));
    EXPECT(std::abs(sin_pi.to_double()) < 1e-14);
    auto sin_half = f80_sin(Float80::from_double(0.5));
    EXPECT(std::abs(sin_half.to_double() - 0.479425538604203) < 1e-13);
}

TEST(f80_cos_basic_angles) {
    auto c0 = f80_cos(Float80::from_double(0.0));
    EXPECT(c0.to_double() == 1.0);
    auto c_half_pi = f80_cos(Float80::from_double(1.57079632679489661923));
    EXPECT(std::abs(c_half_pi.to_double()) < 1e-14);
    auto c_pi = f80_cos(Float80::from_double(3.14159265358979323846));
    EXPECT(std::abs(c_pi.to_double() - (-1.0)) < 1e-14);
}

TEST(f80_2xm1_endpoints) {
    auto r0 = f80_2xm1(Float80::from_double(0.0));
    EXPECT(std::abs(r0.to_double()) < 1e-14);
    auto r1 = f80_2xm1(Float80::from_double(1.0));
    EXPECT(std::abs(r1.to_double() - 1.0) < 1e-5);     // 2^1 - 1 = 1
    auto r_half = f80_2xm1(Float80::from_double(0.5));
    EXPECT(std::abs(r_half.to_double() - 0.41421356237) < 1e-5);
}

TEST(f80_yl2x_basic) {
    auto r = f80_yl2x(Float80::from_double(1.0), Float80::from_double(8.0));
    EXPECT(std::abs(r.to_double() - 3.0) < 1e-10);     // 1 * log2(8) = 3
    auto r2 = f80_yl2x(Float80::from_double(2.0), Float80::from_double(1024.0));
    EXPECT(std::abs(r2.to_double() - 20.0) < 1e-9);    // 2 * log2(1024) = 20
}

TEST(f80_scale_exponent_bump) {
    auto r = f80_scale(Float80::from_double(1.5), Float80::from_double(3.0));
    EXPECT(r.to_double() == 12.0);                     // 1.5 * 2^3 = 12
    auto r2 = f80_scale(Float80::from_double(7.0), Float80::from_double(-2.0));
    EXPECT(r2.to_double() == 1.75);                    // 7 * 2^-2 = 1.75
}

TEST(f80_mul_then_div_roundtrip) {
    // (a * b) / b should equal a for normal values, modulo 80-bit↔double
    // rounding loss in the inputs.
    auto a = Float80::from_double(123.456);
    auto b = Float80::from_double(7.89);
    auto ab = f80_mul(a, b);
    auto r  = f80_div(ab, b);
    EXPECT(r.to_double() == 123.456);   // bit-exact recovery
}
