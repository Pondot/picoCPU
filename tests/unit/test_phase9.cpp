// Unit tests for Phase 9 features: VEX/YMM, CRC32, AES round, SSE bit shifts,
// LAHF/SAHF, PUSHF/POPF, DR load/store breakpoints.

#include "test_framework.h"

#include "emu/cpu.h"
#include "emu/ir.h"
#include "emu/memory.h"

#include "../../src/emu/handlers/handlers.h"

#include <cstring>

using namespace emu;

namespace {

Insn mk_insn(OpKind k, Handler h, u8 op_size = 16) {
    Insn i{};
    i.kind = k; i.handler = h; i.op_size = op_size;
    return i;
}

} // namespace

// -------- AES round (one-step AESENC matches textbook) -----------------------
TEST(aes_round_inputs_outputs_match_textbook) {
    Cpu c;
    // Use plain test vector (all zero state, all zero key) -- AESENC of zero
    // state with zero key = sub_bytes(shift_rows(0)) = sub_bytes(0) = all 0x63.
    Insn i = mk_insn(OpKind::Aesenc, &handlers::op_aesenc);
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0;
    i.src.kind = OperandKind::Xmm; i.src.reg = 1;
    std::memset(c.xmm(0), 0, 16);
    std::memset(c.xmm(1), 0, 16);
    i.handler(c, i);
    for (int j = 0; j < 16; ++j) EXPECT_EQ((int)c.xmm(0)[j], 0x63);
}

// -------- CRC32 against a known reference ------------------------------------
TEST(crc32_known_vector) {
    Cpu c;
    Insn i = mk_insn(OpKind::Crc32, &handlers::op_crc32, 1);
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RCX;
    c.set_r64(reg::RAX, 0);
    c.set_r64(reg::RCX, 'a');
    i.handler(c, i);
    // CRC32 instruction (Castagnoli poly 0x82F63B78 reflected, init=0, no final XOR):
    // CRC32(0, 0x61) = 0x93AD1061. This is the *raw* hardware value, not the
    // FF-init/FF-XOR-wrapped checksum value (0xC1D04330) that protocols use.
    EXPECT_HEX(c.r(reg::RAX), 0x93AD1061ull);
}

// -------- LAHF / SAHF round-trip --------------------------------------------
TEST(lahf_sahf_roundtrip) {
    Cpu c;
    c.set_cf(true);
    c.set_zf(true);
    c.set_pf(true);
    c.set_sf(true);
    c.set_af(true);

    Insn lahf = mk_insn(OpKind::Lahf, &handlers::op_lahf);
    Insn sahf = mk_insn(OpKind::Sahf, &handlers::op_sahf);

    lahf.handler(c, lahf);
    const u8 ah = static_cast<u8>((c.r(reg::RAX) >> 8) & 0xFF);
    EXPECT((ah & 0x01) != 0);    // CF
    EXPECT((ah & 0x40) != 0);    // ZF
    EXPECT((ah & 0x04) != 0);    // PF
    EXPECT((ah & 0x80) != 0);    // SF

    // clear flags, then SAHF should restore them from AH.
    c.set_cf(false); c.set_zf(false); c.set_pf(false); c.set_sf(false); c.set_af(false);
    sahf.handler(c, sahf);
    EXPECT(c.cf()); EXPECT(c.zf()); EXPECT(c.pf()); EXPECT(c.sf());
}

// -------- DR0 execute breakpoint match --------------------------------------
TEST(dr0_execute_breakpoint_match) {
    Cpu c;
    c.set_dr(0, 0x1234);
    // DR7: local enable for DR0 (bit 0), condition 00 (execute), length 00 (1).
    c.set_dr(7, 0x1ull);
    EXPECT(c.exec_breakpoint_at(0x1234));
    EXPECT(!c.exec_breakpoint_at(0x1235));
    EXPECT(!c.exec_breakpoint_at(0x4321));
}

// -------- DR1 store-only breakpoint covers a range --------------------------
TEST(dr1_store_breakpoint_with_length) {
    Cpu c;
    c.set_dr(1, 0x2000);
    // DR7: enable DR1 (bits 2-3 set local), condition for DR1 = 01 (store, bits 20-21), length = 01 (2 bytes, bits 22-23).
    // 1 << 2 | (0x01 << 20) | (0x01 << 22)
    c.set_dr(7, (1ull << 2) | (1ull << 20) | (1ull << 22));
    EXPECT(c.data_breakpoint_at(0x2000, /*write*/true, 1));
    EXPECT(c.data_breakpoint_at(0x2001, /*write*/true, 1));
    // load to the same range should NOT trigger (cond=01 = store-only).
    EXPECT(!c.data_breakpoint_at(0x2000, /*write*/false, 1));
}

// -------- VPXOR-128 zeros upper YMM -----------------------------------------
TEST(vpxor_128_zeros_ymm_upper) {
    Cpu c;
    // Pre-fill ymm0 upper half with non-zero so we can detect zeroing.
    for (int i = 16; i < 32; ++i) c.ymm(0)[i] = 0xAA;
    // Pre-fill ymm1 and ymm2 with known values.
    for (int i = 0; i < 16; ++i) { c.ymm(1)[i] = 0xFF; c.ymm(2)[i] = 0x0F; }

    Insn i = mk_insn(OpKind::Vpxor, &handlers::op_vpxor, 16);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;   // vvvv = ymm1
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);

    for (int j = 0; j < 16; ++j) EXPECT_EQ((int)c.ymm(0)[j], 0xF0);
    // Upper half must have been zeroed by VEX-128.
    for (int j = 16; j < 32; ++j) EXPECT_EQ((int)c.ymm(0)[j], 0);
}

// -------- BMI1/2 -------------------------------------------------------------

TEST(andn_basic) {
    Cpu c;
    Insn i = mk_insn(OpKind::Andn, &handlers::op_andn, 8);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX; i.dst.index = reg::RCX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RDX;
    c.set_r64(reg::RCX, 0x0F0F0F0F0F0F0F0Full);
    c.set_r64(reg::RDX, 0xFFFFFFFFFFFFFFFFull);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0xF0F0F0F0F0F0F0F0ull);
}

TEST(blsi_isolates_low_set_bit) {
    Cpu c;
    Insn i = mk_insn(OpKind::Blsi, &handlers::op_blsi, 8);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX; i.dst.index = reg::RAX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RCX;
    c.set_r64(reg::RCX, 0x00FF0000ull);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x00010000ull);
}

TEST(blsr_clears_low_set_bit) {
    Cpu c;
    Insn i = mk_insn(OpKind::Blsr, &handlers::op_blsr, 8);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX; i.dst.index = reg::RAX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RCX;
    c.set_r64(reg::RCX, 0x12345670ull);
    i.handler(c, i);
    // Low set bit of 0x...70 is bit 4 -> clear it -> 0x12345660
    EXPECT_HEX(c.r(reg::RAX), 0x12345660ull);
}

TEST(bextr_extracts_bits) {
    Cpu c;
    Insn i = mk_insn(OpKind::Bextr, &handlers::op_bextr, 8);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX; i.dst.index = reg::RCX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RDX;
    c.set_r64(reg::RCX, (4ull) | (8ull << 8));     // start=4, len=8
    c.set_r64(reg::RDX, 0x123456789ABCDEFFull);
    i.handler(c, i);
    // (0x...DEFF >> 4) = 0x...DEF, & 0xFF = 0xEF
    EXPECT_HEX(c.r(reg::RAX), 0xEFull);
}

TEST(rorx_rotates_no_flag_change) {
    Cpu c;
    c.set_cf(true);
    Insn i = mk_insn(OpKind::Rorx, &handlers::op_rorx, 8);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Reg; i.dst.reg = reg::RAX;
    i.src.kind = OperandKind::Reg; i.src.reg = reg::RCX;
    i.imm_extra = 4;
    c.set_r64(reg::RCX, 0xFEDCBA9876543210ull);
    i.handler(c, i);
    EXPECT_HEX(c.r(reg::RAX), 0x0FEDCBA987654321ull);
    EXPECT(c.cf());     // RORX leaves flags unchanged
}

TEST(pext_pdep_roundtrip) {
    Cpu c;
    // PEXT: with mask = 0xFF00, src = 0xABCD, pull bits at mask positions
    //   bits 15..8 of src = 0xAB -> low 8 bits of result = 0xAB
    Insn ix = mk_insn(OpKind::Pext, &handlers::op_pext, 8);
    ix.flags = INSN_FLAG_VEX;
    ix.dst.kind = OperandKind::Reg; ix.dst.reg = reg::RAX; ix.dst.index = reg::RCX;
    ix.src.kind = OperandKind::Reg; ix.src.reg = reg::RDX;
    c.set_r64(reg::RCX, 0xABCDull);
    c.set_r64(reg::RDX, 0xFF00ull);
    ix.handler(c, ix);
    EXPECT_HEX(c.r(reg::RAX), 0xABull);

    // PDEP: scatter 0xAB into positions 15..8 of mask 0xFF00
    Insn id = mk_insn(OpKind::Pdep, &handlers::op_pdep, 8);
    id.flags = INSN_FLAG_VEX;
    id.dst.kind = OperandKind::Reg; id.dst.reg = reg::RBX; id.dst.index = reg::RAX;
    id.src.kind = OperandKind::Reg; id.src.reg = reg::RDX;
    id.handler(c, id);
    EXPECT_HEX(c.r(reg::RBX), 0xAB00ull);
}

// -------- PSLLD by 4 ---------------------------------------------------------
TEST(pslld_shifts_each_lane) {
    Cpu c;
    const u32 lanes_in[4]  = {1u, 2u, 0x80000000u, 0xFFu};
    for (int i = 0; i < 4; ++i) std::memcpy(c.xmm(0) + 4*i, &lanes_in[i], 4);

    Insn i = mk_insn(OpKind::Pslld, &handlers::op_pslld, 16);
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0;
    i.src.kind = OperandKind::Imm; i.src.imm = 4;
    i.imm_extra = 4;       // shift count for imm form
    i.handler(c, i);

    const u32 expect[4] = {16u, 32u, 0u, 0xFF0u};
    for (int k = 0; k < 4; ++k) {
        u32 v; std::memcpy(&v, c.xmm(0) + 4*k, 4);
        EXPECT_HEX(v, expect[k]);
    }
}
