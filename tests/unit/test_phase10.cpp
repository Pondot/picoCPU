// Unit tests for Phase 10: VEX-3op SSE, SHA-NI, PCMPESTR/ISTR, memory
// protection, snapshot/restore.

#include "test_framework.h"

#include "emu/cpu.h"
#include "emu/ir.h"
#include "emu/memory.h"
#include "emu/shadow_pages.h"

#include "../../src/emu/handlers/handlers.h"

#include <cstring>

using namespace emu;

namespace {

Insn mk(OpKind k, Handler h, u8 size = 16) {
    Insn i{};
    i.kind = k; i.handler = h; i.op_size = size;
    return i;
}

} // namespace

// VPSHUFB with vvvv source -- output bytes pulled from src1 (vvvv) indexed by
// the low nibble of src2 (insn.src) bytes; MSB-set bytes zero out.
TEST(vpshufb_uses_vvvv_as_src1) {
    Cpu c;
    // ymm[1] = identity 0..15  (this is vvvv)
    for (int i = 0; i < 16; ++i) c.ymm(1)[i] = static_cast<u8>(i);
    // ymm[2] = shuffle control: reverse low 8, zero the rest via MSB-set
    for (int i = 0; i < 8; ++i)   c.ymm(2)[i] = static_cast<u8>(7 - i);
    for (int i = 8; i < 16; ++i)  c.ymm(2)[i] = 0x80;
    // ymm[0] = something non-zero in upper half so we can verify VEX-128 zeroing
    for (int i = 16; i < 32; ++i) c.ymm(0)[i] = 0xCC;

    Insn i = mk(OpKind::Vpshufb, &handlers::op_vpshufb, 16);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;  // vvvv = ymm1
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);

    // Low 8 bytes: reversed 0..7  ->  7,6,5,4,3,2,1,0
    for (int k = 0; k < 8; ++k) EXPECT_EQ((int)c.ymm(0)[k], 7 - k);
    // High 8 bytes: zero (MSB-set in control)
    for (int k = 8; k < 16; ++k) EXPECT_EQ((int)c.ymm(0)[k], 0);
    // Upper YMM: zero (VEX-128 zeroing)
    for (int k = 16; k < 32; ++k) EXPECT_EQ((int)c.ymm(0)[k], 0);
}

// VAESENC with vvvv = state, src = key. Both zero -> AESENC(0,0) -> SBOX[0]*16 = 0x63 every byte.
TEST(vaesenc_zero_state_zero_key) {
    Cpu c;
    std::memset(c.ymm(1), 0, 16);   // vvvv
    std::memset(c.ymm(2), 0, 16);   // key
    Insn i = mk(OpKind::Vaesenc, &handlers::op_vaesenc);
    i.flags = INSN_FLAG_VEX;
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0; i.dst.index = 1;
    i.src.kind = OperandKind::Xmm; i.src.reg = 2;
    i.handler(c, i);
    // SubBytes(ShiftRows(zero)) ^ MixColumns(...) ^ zero key.
    // After ShiftRows of all-zero state -> still zero. SubBytes(0) = 0x63 everywhere.
    // MixColumns of constant 0x63 columns: each column has 4 identical bytes,
    // so MixColumns result is also 4 identical bytes per column. Computing:
    //   for byte b: 2*b ^ 3*b ^ b ^ b = (2^3)*b ^ 2b = 1*b XOR_PROD... actually
    //   for identical column [b,b,b,b], MixColumns gives:
    //     out0 = 2b ^ 3b ^ b ^ b  -> wait MixColumns matrix is:
    //       [2 3 1 1]
    //       [1 2 3 1]
    //       [1 1 2 3]
    //       [3 1 1 2]
    //   For (b,b,b,b): out0 = 2b ^ 3b ^ 1b ^ 1b = (2 ^ 3 ^ 1 ^ 1)b = 1*b = b.
    //   So MixColumns of constant column = same column = 0x63.
    // Then XOR with zero key -> all 0x63.
    for (int k = 0; k < 16; ++k) EXPECT_EQ((int)c.ymm(0)[k], 0x63);
}

// PCMPISTRI: implicit lengths, find first matching byte index. ECX = index.
// Control byte 0x0C = byte/equal-each/no-polarity/index-LSB.
TEST(pcmpistri_equal_each_finds_first_match) {
    Cpu c;
    // a = "hello\0..."  (lane 0 = 'h', etc.)
    const char a[16] = {'h','e','l','l','o',0,0,0,0,0,0,0,0,0,0,0};
    const char b[16] = {'?','?','l','?','?','?','?','?',0,0,0,0,0,0,0,0};
    std::memcpy(c.xmm(0), a, 16);
    std::memcpy(c.xmm(1), b, 16);

    Insn i = mk(OpKind::Pcmpistri, &handlers::op_pcmpistri);
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0;
    i.src.kind = OperandKind::Xmm; i.src.reg = 1;
    i.imm_extra = 0x0C;     // byte, equal-each, no polarity, index LSB-first
    i.handler(c, i);

    // Equal-each: a[i] == b[i]. Match at lane 2 ('l' == 'l').
    EXPECT_EQ((int)c.r(reg::RCX), 2);
}

// PCMPISTRI: equal-any mode -- for each byte of B, does any byte of A match?
// (Mode 1, ctrl bits [3:2] = 01 -> 0x04)
TEST(pcmpistri_equal_any) {
    Cpu c;
    // a = "abc\0..."  ; b = "xby\0..."
    const char a[16] = {'a','b','c', 0,0,0,0,0,0,0,0,0,0,0,0,0};
    const char b[16] = {'x','b','y', 0,0,0,0,0,0,0,0,0,0,0,0,0};
    std::memcpy(c.xmm(0), a, 16);
    std::memcpy(c.xmm(1), b, 16);

    Insn i = mk(OpKind::Pcmpistri, &handlers::op_pcmpistri);
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0;
    i.src.kind = OperandKind::Xmm; i.src.reg = 1;
    i.imm_extra = 0x04;     // byte, equal-any, no polarity, index LSB-first
    i.handler(c, i);

    // Mask: lane 1 ('b' in B is in A) -- index = 1.
    EXPECT_EQ((int)c.r(reg::RCX), 1);
}

// SHA1MSG2 should produce W19 = rotl(W3 ^ W19_prev_xor_via_msg1, 1) ... the
// SDM semantics. Just check the function runs without faulting and produces
// non-zero output for non-zero input.
TEST(sha1msg2_smoke) {
    Cpu c;
    for (int i = 0; i < 16; ++i) c.ymm(0)[i] = static_cast<u8>(i + 1);
    for (int i = 0; i < 16; ++i) c.ymm(1)[i] = static_cast<u8>(i ^ 0x55);
    Insn i = mk(OpKind::Sha1msg2, &handlers::op_sha1msg2);
    i.dst.kind = OperandKind::Xmm; i.dst.reg = 0;
    i.src.kind = OperandKind::Xmm; i.src.reg = 1;
    i.handler(c, i);
    // Non-zero output; this just smoke-tests the pipeline.
    bool any_nonzero = false;
    for (int k = 0; k < 16; ++k) if (c.ymm(0)[k]) { any_nonzero = true; break; }
    EXPECT(any_nonzero);
}

// Memory protection: write to a region marked read-only faults.
TEST(memory_protect_read_only_blocks_write) {
    ShadowPages sp(nullptr);
    sp.add_private_region(0x10000, 0x1000);
    sp.protect(0x10000, 0x1000, ShadowPages::PROT_R);   // RO

    u8 buf = 0x42;
    Status s = sp.write(0x10800, 1, &buf);
    EXPECT_EQ((int)s, (int)Status::ProtectionViolation);

    // A separate region without prot is still writable.
    sp.add_private_region(0x20000, 0x1000);
    sp.protect(0x20000, 0x1000, ShadowPages::PROT_R | ShadowPages::PROT_W);
    EXPECT_EQ((int)sp.write(0x20800, 1, &buf), (int)Status::Ok);
}

// Cpu snapshot then restore.
TEST(cpu_snapshot_and_restore_roundtrip) {
    Cpu c;
    c.set_r64(reg::RAX, 0xDEADBEEFCAFEBABEull);
    c.set_r64(reg::RCX, 0x1234567890ABCDEFull);
    c.set_rip(0x4000);
    c.set_cf(true); c.set_zf(false); c.set_sf(true);
    for (int i = 0; i < 16; ++i) c.ymm(0)[i] = static_cast<u8>(i ^ 0xA);

    Cpu::Snapshot s;
    c.snapshot(s);

    // Mutate.
    c.set_r64(reg::RAX, 0);
    c.set_rip(0);
    c.set_cf(false);
    std::memset(c.ymm(0), 0xFF, 16);

    // Restore.
    c.restore(s);

    EXPECT_HEX(c.r(reg::RAX), 0xDEADBEEFCAFEBABEull);
    EXPECT_HEX(c.r(reg::RCX), 0x1234567890ABCDEFull);
    EXPECT_HEX(c.rip(), 0x4000ull);
    EXPECT(c.cf());
    for (int i = 0; i < 16; ++i) EXPECT_EQ((int)c.ymm(0)[i], i ^ 0xA);
}
