// Handler unit tests. Each test sets up Cpu state + Insn, calls the handler,
// checks results and (for arithmetic) flag bits.

#include "test_framework.h"

#include "emu/cpu.h"
#include "emu/ir.h"

// Pull the handler declarations from src/emu/handlers/handlers.h via a
// relative include so tests don't need their own include path.
#include "../../src/emu/handlers/handlers.h"

using namespace emu;

namespace {

Insn make_rr(OpKind k, Handler h, u8 dst_reg, u8 src_reg, u8 size = 8) {
    Insn i{};
    i.kind     = k;
    i.handler  = h;
    i.op_size  = size;
    i.len      = 3;
    i.dst.kind = OperandKind::Reg;
    i.dst.reg  = dst_reg;
    i.src.kind = OperandKind::Reg;
    i.src.reg  = src_reg;
    return i;
}

Insn make_ri(OpKind k, Handler h, u8 dst_reg, i64 imm, u8 size = 8) {
    Insn i{};
    i.kind     = k;
    i.handler  = h;
    i.op_size  = size;
    i.len      = 4;
    i.dst.kind = OperandKind::Reg;
    i.dst.reg  = dst_reg;
    i.src.kind = OperandKind::Imm;
    i.src.imm  = imm;
    return i;
}

} // namespace

TEST(handler_mov_imm64) {
    Cpu c;
    Insn i = make_ri(OpKind::Mov, &handlers::op_mov, reg::RAX, static_cast<i64>(0xDEADBEEFCAFEBABEull));
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0xDEADBEEFCAFEBABEull);
}

TEST(handler_xor_self_zero) {
    Cpu c;
    c.set_r64(reg::RAX, 0xDEADBEEFCAFEBABEull);
    Insn i = make_rr(OpKind::Xor, &handlers::op_xor, reg::RAX, reg::RAX);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0ull);
    EXPECT(c.zf());
    EXPECT(!c.cf());
    EXPECT(!c.of());
}

TEST(handler_add_carry) {
    Cpu c;
    c.set_r64(reg::RAX, 0xFFFFFFFFFFFFFFFFull);
    c.set_r64(reg::RCX, 1);
    Insn i = make_rr(OpKind::Add, &handlers::op_add, reg::RAX, reg::RCX);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0ull);
    EXPECT(c.zf());
    EXPECT(c.cf());      // overflowed
    EXPECT(!c.of());     // unsigned overflow ≠ signed overflow
}

TEST(handler_sub_signed_overflow) {
    Cpu c;
    c.set_r64(reg::RAX, 0x8000000000000000ull);  // INT64_MIN
    c.set_r64(reg::RCX, 1);
    Insn i = make_rr(OpKind::Sub, &handlers::op_sub, reg::RAX, reg::RCX);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x7FFFFFFFFFFFFFFFull);
    EXPECT(c.of());       // INT64_MIN - 1 overflows
    EXPECT(!c.cf());      // unsigned: 0x80.. - 1 doesn't borrow
}

TEST(handler_imul_2op_match) {
    // Mirror the IMUL step in the mixer:  rax * rcx (signed, low 64 bits).
    Cpu c;
    c.set_r64(reg::RAX, 0x3ebe49e03b1887baull);
    c.set_r64(reg::RCX, 0xbf58476d1ce4e5b9ull);
    Insn i = make_rr(OpKind::Imul, &handlers::op_imul, reg::RAX, reg::RCX);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x46caa8cadfcb776aull);
}

TEST(handler_ror_match) {
    // ROR rax, 47 -- matches mixer step.
    Cpu c;
    c.set_r64(reg::RAX, 0x43dd1f5f24f01d8cull);
    Insn i = make_ri(OpKind::Ror, &handlers::op_ror, reg::RAX, 47, 8);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x3ebe49e03b1887baull);
}

TEST(handler_shr_match) {
    // SHR rcx, 30 -- matches mixer step.
    Cpu c;
    c.set_r64(reg::RCX, 0x46caa8cadfcb776aull);
    Insn i = make_ri(OpKind::Shr, &handlers::op_shr, reg::RCX, 30, 8);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RCX), 0x11b2aa32bull);
}

TEST(handler_and_clears_cf_of) {
    Cpu c;
    c.set_r64(reg::RAX, 0xFFFFFFFFFFFFFFFFull);
    c.set_r64(reg::RCX, 0x00FF00FF00FF00FFull);
    c.set_cf(true); c.set_of(true);
    Insn i = make_rr(OpKind::And, &handlers::op_and, reg::RAX, reg::RCX);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x00FF00FF00FF00FFull);
    EXPECT(!c.cf());
    EXPECT(!c.of());
}

TEST(handler_inc_preserves_cf) {
    Cpu c;
    c.set_cf(true);
    c.set_r64(reg::RAX, 1);
    Insn i{};
    i.kind = OpKind::Inc; i.handler = &handlers::op_inc; i.op_size = 8;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX;
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 2ull);
    EXPECT(c.cf());     // INC preserves CF
}

TEST(handler_cmp_eq_sets_zf) {
    Cpu c;
    c.set_r64(reg::RAX, 0x42);
    c.set_r64(reg::RCX, 0x42);
    Insn i = make_rr(OpKind::Cmp, &handlers::op_cmp, reg::RAX, reg::RCX);
    i.handler(c, i);
    EXPECT(c.zf());
    EXPECT(!c.cf());
    // rax unchanged
    EXPECT_HEX(c.r(reg::RAX), 0x42ull);
}

TEST(handler_shl_overflow_one) {
    Cpu c;
    c.set_r64(reg::RAX, 0x4000000000000000ull);
    Insn i = make_ri(OpKind::Shl, &handlers::op_shl, reg::RAX, 1, 8);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x8000000000000000ull);
    EXPECT(!c.cf());          // bit shifted out is 0
    EXPECT(c.of());           // SHL 1: OF = result-sign XOR CF = 1
}

TEST(handler_movzx_byte_to_qword) {
    Cpu c;
    c.set_r64(reg::RCX, 0xDEADBEEFCAFEBA80ull);
    Insn i{};
    i.kind = OpKind::MovZx; i.handler = &handlers::op_movzx;
    i.op_size = 8;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RCX;
    i.src.scale = 1;   // decoder stashes source size here
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x80ull);
}

TEST(handler_movsx_byte_to_qword) {
    Cpu c;
    c.set_r64(reg::RCX, 0xDEADBEEFCAFEBA80ull);  // low byte = 0x80, negative
    Insn i{};
    i.kind = OpKind::MovSx; i.handler = &handlers::op_movsx;
    i.op_size = 8;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RCX;
    i.src.scale = 1;
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0xFFFFFFFFFFFFFF80ull);  // sign-extended
}
