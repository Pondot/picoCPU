// Decoder unit tests. We hand-write byte sequences (matching what MSVC emits
// for the mixer) and check the produced Insn.

#include "test_framework.h"

#include "emu/decoder.h"
#include "emu/ir.h"

using namespace emu;

static Insn decode(const std::initializer_list<u8>& bytes) {
    Insn out{};
    std::vector<u8> v(bytes);
    Status s = decode_bytes(v.data(), v.size(), 0, out);
    if (fail(s)) {
        std::printf("  (decode failed: %.*s)\n",
                    static_cast<int>(to_string(s).size()),
                    to_string(s).data());
    }
    return out;
}

TEST(decode_mov_rr64) {
    // 48 8b c1  ->  mov rax, rcx
    Insn i = decode({0x48, 0x8B, 0xC1});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Mov));
    EXPECT_EQ(static_cast<int>(i.op_size), 8);
    EXPECT_EQ(static_cast<int>(i.len), 3);
    EXPECT_EQ(static_cast<int>(i.dst.kind), static_cast<int>(OperandKind::Reg));
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RAX));
    EXPECT_EQ(static_cast<int>(i.src.kind), static_cast<int>(OperandKind::Reg));
    EXPECT_EQ(static_cast<int>(i.src.reg), static_cast<int>(reg::RCX));
}

TEST(decode_mov_imm64) {
    // 48 b9 a5 a5 a5 a5 a5 a5 a5 a5  ->  mov rcx, 0xa5a5a5a5a5a5a5a5
    Insn i = decode({0x48, 0xB9, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Mov));
    EXPECT_EQ(static_cast<int>(i.op_size), 8);
    EXPECT_EQ(static_cast<int>(i.len), 10);
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RCX));
    EXPECT_HEX(static_cast<unsigned long long>(static_cast<u64>(i.src.imm)),
               0xA5A5A5A5A5A5A5A5ull);
}

TEST(decode_xor_rr64) {
    // 48 33 c1  ->  xor rax, rcx
    Insn i = decode({0x48, 0x33, 0xC1});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Xor));
    EXPECT_EQ(static_cast<int>(i.op_size), 8);
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RAX));
    EXPECT_EQ(static_cast<int>(i.src.reg), static_cast<int>(reg::RCX));
}

TEST(decode_sub_rr64) {
    // 48 2b c1  ->  sub rax, rcx
    Insn i = decode({0x48, 0x2B, 0xC1});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Sub));
}

TEST(decode_imul_2op) {
    // 48 0f af c1  ->  imul rax, rcx
    Insn i = decode({0x48, 0x0F, 0xAF, 0xC1});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Imul));
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RAX));
    EXPECT_EQ(static_cast<int>(i.src.reg), static_cast<int>(reg::RCX));
}

TEST(decode_ror_imm8) {
    // 48 c1 c8 2f  ->  ror rax, 0x2f
    Insn i = decode({0x48, 0xC1, 0xC8, 0x2F});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Ror));
    EXPECT_EQ(static_cast<int>(i.op_size), 8);
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RAX));
    EXPECT_EQ(static_cast<int>(i.src.kind), static_cast<int>(OperandKind::Imm));
    EXPECT_EQ(i.src.imm, 0x2F);
}

TEST(decode_shr_imm8) {
    // 48 c1 e9 1e  ->  shr rcx, 0x1e
    Insn i = decode({0x48, 0xC1, 0xE9, 0x1E});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Shr));
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RCX));
}

TEST(decode_ret_near) {
    Insn i = decode({0xC3});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Ret));
    EXPECT_EQ(static_cast<int>(i.len), 1);
    EXPECT(i.flags & INSN_FLAG_CONTROL_FLOW);
}

TEST(decode_jz_short) {
    // 74 10  ->  jz +0x10
    Insn i = decode({0x74, 0x10});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Jcc));
    EXPECT_EQ(static_cast<int>(i.cond), static_cast<int>(Cc::Z));
    EXPECT_EQ(static_cast<int>(i.len), 2);
    EXPECT_EQ(i.dst.imm, 0x10);
}

TEST(decode_lea_riprel) {
    // 48 8d 05 11 22 33 44  ->  lea rax, [rip+0x44332211]
    Insn i = decode({0x48, 0x8D, 0x05, 0x11, 0x22, 0x33, 0x44});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Lea));
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RAX));
    EXPECT_EQ(static_cast<int>(i.src.kind), static_cast<int>(OperandKind::RipRelMem));
    EXPECT_EQ(i.src.imm, 0x44332211);
}

TEST(decode_add_rm_imm8_sx) {
    // 48 83 c0 05  ->  add rax, 5
    Insn i = decode({0x48, 0x83, 0xC0, 0x05});
    EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Add));
    EXPECT_EQ(static_cast<int>(i.op_size), 8);
    EXPECT_EQ(static_cast<int>(i.dst.kind), static_cast<int>(OperandKind::Reg));
    EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RAX));
    EXPECT_EQ(i.src.imm, 5);
}

TEST(decode_push_pop_r64) {
    // 50  ->  push rax
    {
        Insn i = decode({0x50});
        EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Push));
        EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::RAX));
    }
    // 41 58  ->  pop r8
    {
        Insn i = decode({0x41, 0x58});
        EXPECT_EQ(static_cast<int>(i.kind), static_cast<int>(OpKind::Pop));
        EXPECT_EQ(static_cast<int>(i.dst.reg), static_cast<int>(reg::R8));
    }
}
