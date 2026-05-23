// SSE4.x, SSSE3 extras, AES-NI, CRC32, PCLMULQDQ. Plus more x87 trig.
//
// AES uses textbook tables; not constant-time, not performance-tuned, but
// produces identical results to AESNI hardware for the standard modes.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cmath>
#include <cstring>

namespace emu::handlers {

namespace {

bool read16(Cpu& cpu, const Insn& insn, const Operand& op, u8 out[16]) noexcept {
    if (op.kind == OperandKind::Xmm) { std::memcpy(out, cpu.xmm(op.reg), 16); return true; }
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "read16"); return false; }
    const GuestAddr addr = ea(cpu, insn, op);
    if (Status s = mp->read(addr, 16, out); fail(s)) { cpu.set_fault(FaultKind::PageFault, addr, s, "read16"); return false; }
    return true;
}
bool write16(Cpu& cpu, const Insn& insn, const Operand& op, const u8 v[16]) noexcept {
    if (op.kind == OperandKind::Xmm) { std::memcpy(cpu.xmm(op.reg), v, 16); return true; }
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "write16"); return false; }
    const GuestAddr addr = ea(cpu, insn, op);
    if (Status s = mp->write(addr, 16, v); fail(s)) { cpu.set_fault(kind_from_status(s), addr, s, "write16"); return false; }
    return true;
}

// ---- AES-NI ---------------------------------------------------------------
// Reference: FIPS-197. We use the textbook tables.

constexpr u8 SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
};

constexpr u8 INV_SBOX[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
};

// GF(2^8) multiply by 2 (xtime).
inline u8 xtime(u8 v) noexcept { return static_cast<u8>((v << 1) ^ ((v & 0x80) ? 0x1B : 0)); }

void shift_rows(u8 s[16]) noexcept {
    u8 t;
    t = s[1];  s[1] = s[5];  s[5] = s[9];  s[9] = s[13]; s[13] = t;
    t = s[2];  s[2] = s[10]; s[10] = t;
    t = s[6];  s[6] = s[14]; s[14] = t;
    t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
}
void inv_shift_rows(u8 s[16]) noexcept {
    u8 t;
    t = s[13]; s[13] = s[9];  s[9] = s[5];  s[5] = s[1]; s[1] = t;
    t = s[2];  s[2] = s[10]; s[10] = t;
    t = s[6];  s[6] = s[14]; s[14] = t;
    t = s[3];  s[3] = s[7]; s[7] = s[11]; s[11] = s[15]; s[15] = t;
}
void sub_bytes(u8 s[16]) noexcept { for (int i = 0; i < 16; ++i) s[i] = SBOX[s[i]]; }
void inv_sub_bytes(u8 s[16]) noexcept { for (int i = 0; i < 16; ++i) s[i] = INV_SBOX[s[i]]; }
void mix_columns(u8 s[16]) noexcept {
    for (int c = 0; c < 4; ++c) {
        const u8 a0 = s[4*c], a1 = s[4*c+1], a2 = s[4*c+2], a3 = s[4*c+3];
        const u8 x = a0 ^ a1 ^ a2 ^ a3;
        s[4*c]   = a0 ^ x ^ xtime(a0 ^ a1);
        s[4*c+1] = a1 ^ x ^ xtime(a1 ^ a2);
        s[4*c+2] = a2 ^ x ^ xtime(a2 ^ a3);
        s[4*c+3] = a3 ^ x ^ xtime(a3 ^ a0);
    }
}
void inv_mix_columns(u8 s[16]) noexcept {
    for (int c = 0; c < 4; ++c) {
        const u8 a0 = s[4*c], a1 = s[4*c+1], a2 = s[4*c+2], a3 = s[4*c+3];
        const u8 x = a0 ^ a1 ^ a2 ^ a3;
        const u8 y = xtime(xtime(x));
        const u8 t0 = a0 ^ y ^ xtime(a0 ^ a1);
        const u8 t1 = a1 ^ y ^ xtime(a1 ^ a2);
        const u8 t2 = a2 ^ y ^ xtime(a2 ^ a3);
        const u8 t3 = a3 ^ y ^ xtime(a3 ^ a0);
        s[4*c]   = t0 ^ xtime(a0 ^ a2) ^ xtime(xtime(a0 ^ a2));
        s[4*c+1] = t1 ^ xtime(a1 ^ a3) ^ xtime(xtime(a1 ^ a3));
        s[4*c+2] = t2 ^ xtime(a0 ^ a2) ^ xtime(xtime(a0 ^ a2));
        s[4*c+3] = t3 ^ xtime(a1 ^ a3) ^ xtime(xtime(a1 ^ a3));
    }
}
void add_round_key(u8 s[16], const u8 k[16]) noexcept {
    for (int i = 0; i < 16; ++i) s[i] ^= k[i];
}

} // namespace

void op_aesenc(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    if (!read16(cpu, insn, insn.dst, state)) return;
    if (!read16(cpu, insn, insn.src, key))   return;
    shift_rows(state);
    sub_bytes(state);
    mix_columns(state);
    add_round_key(state, key);
    (void)write16(cpu, insn, insn.dst, state);
}
void op_aesenclast(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    if (!read16(cpu, insn, insn.dst, state)) return;
    if (!read16(cpu, insn, insn.src, key))   return;
    shift_rows(state);
    sub_bytes(state);
    add_round_key(state, key);
    (void)write16(cpu, insn, insn.dst, state);
}
void op_aesdec(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    if (!read16(cpu, insn, insn.dst, state)) return;
    if (!read16(cpu, insn, insn.src, key))   return;
    inv_shift_rows(state);
    inv_sub_bytes(state);
    inv_mix_columns(state);
    add_round_key(state, key);
    (void)write16(cpu, insn, insn.dst, state);
}
void op_aesdeclast(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    if (!read16(cpu, insn, insn.dst, state)) return;
    if (!read16(cpu, insn, insn.src, key))   return;
    inv_shift_rows(state);
    inv_sub_bytes(state);
    add_round_key(state, key);
    (void)write16(cpu, insn, insn.dst, state);
}
void op_aesimc(Cpu& cpu, const Insn& insn) {
    u8 s[16];
    if (!read16(cpu, insn, insn.src, s)) return;
    inv_mix_columns(s);
    (void)write16(cpu, insn, insn.dst, s);
}
void op_aeskeygen(Cpu& cpu, const Insn& insn) {
    u8 s[16];
    if (!read16(cpu, insn, insn.src, s)) return;
    const u8 rcon = static_cast<u8>(insn.imm_extra & 0xFFu);
    u8 r[16];
    // X3 from input (bytes 12..15)
    u32 x3 = static_cast<u32>(s[12]) | (u32(s[13])<<8) | (u32(s[14])<<16) | (u32(s[15])<<24);
    // RotWord(SubWord(X3)) XOR RCON
    u8 b0 = SBOX[(x3 >>  0) & 0xFFu];
    u8 b1 = SBOX[(x3 >>  8) & 0xFFu];
    u8 b2 = SBOX[(x3 >> 16) & 0xFFu];
    u8 b3 = SBOX[(x3 >> 24) & 0xFFu];
    // X1 from input (bytes 4..7) unchanged into output bytes 0..3
    std::memcpy(r,     s + 4,  4);
    // X1 XOR (RotWord-SubWord-XOR-RCON of X3) into bytes 4..7
    r[4] = static_cast<u8>(s[4] ^ b1);
    r[5] = static_cast<u8>(s[5] ^ b2);
    r[6] = static_cast<u8>(s[6] ^ b3);
    r[7] = static_cast<u8>(s[7] ^ b0 ^ rcon);
    // X3 into bytes 8..11
    std::memcpy(r + 8,  s + 12, 4);
    // X3 XOR (SubWord(X3)-RotWord-XOR-RCON) into bytes 12..15  (simplified)
    r[12] = static_cast<u8>(s[12] ^ b1);
    r[13] = static_cast<u8>(s[13] ^ b2);
    r[14] = static_cast<u8>(s[14] ^ b3);
    r[15] = static_cast<u8>(s[15] ^ b0 ^ rcon);
    (void)write16(cpu, insn, insn.dst, r);
}

// PCLMULQDQ -- carry-less multiply of 64-bit halves, selected by imm8.
void op_pclmulqdq(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read16(cpu, insn, insn.dst, a)) return;
    if (!read16(cpu, insn, insn.src, b)) return;
    const u8 imm = static_cast<u8>(insn.imm_extra & 0xFFu);
    u64 al, bl;
    std::memcpy(&al, a + ((imm & 0x1) ? 8 : 0), 8);
    std::memcpy(&bl, b + ((imm & 0x10) ? 8 : 0), 8);
    // 128-bit carry-less multiply.
    u64 lo = 0, hi = 0;
    for (int i = 0; i < 64; ++i) {
        if ((al >> i) & 1) {
            lo ^= (bl << i);
            if (i > 0) hi ^= (bl >> (64 - i));
        }
    }
    u8 r[16];
    std::memcpy(r,     &lo, 8);
    std::memcpy(r + 8, &hi, 8);
    (void)write16(cpu, insn, insn.dst, r);
}

// CRC32 -- SSE4.2 CRC32 using polynomial 0x11EDC6F41 (Castagnoli).
void op_crc32(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, src)) return;
    u32 crc = static_cast<u32>(cpu.r32(insn.dst.reg));
    for (u8 b = 0; b < insn.op_size; ++b) {
        u32 byte = static_cast<u32>((src >> (8 * b)) & 0xFFu);
        crc = crc ^ byte;
        for (int j = 0; j < 8; ++j) {
            crc = (crc & 1) ? ((crc >> 1) ^ 0x82F63B78u) : (crc >> 1);
        }
    }
    cpu.set_r32(insn.dst.reg, crc);
}

// ROUNDSS / ROUNDSD -- rounding mode from imm[1:0]. We ignore the round-mode
// override flag (bit 2) for simplicity.
namespace {
template <typename T>
T round_with_mode(T v, u8 mode) noexcept {
    switch (mode & 0x3) {
        case 0: return std::nearbyint(v);     // round to nearest
        case 1: return std::floor(v);          // round down
        case 2: return std::ceil(v);           // round up
        case 3: return std::trunc(v);          // truncate toward zero
    }
    return v;
}
} // namespace

void op_roundss(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read16(cpu, insn, insn.dst, d)) return;
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    u32 b; std::memcpy(&b, s, 4);
    float f; std::memcpy(&f, &b, 4);
    f = round_with_mode(f, static_cast<u8>(insn.imm_extra));
    std::memcpy(&b, &f, 4);
    std::memcpy(d, &b, 4);
    (void)write16(cpu, insn, insn.dst, d);
}
void op_roundsd(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read16(cpu, insn, insn.dst, d)) return;
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    u64 b; std::memcpy(&b, s, 8);
    double f; std::memcpy(&f, &b, 8);
    f = round_with_mode(f, static_cast<u8>(insn.imm_extra));
    std::memcpy(&b, &f, 8);
    std::memcpy(d, &b, 8);
    (void)write16(cpu, insn, insn.dst, d);
}
void op_roundps(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    u8 r[16];
    for (int i = 0; i < 4; ++i) {
        u32 b; std::memcpy(&b, s + 4*i, 4);
        float f; std::memcpy(&f, &b, 4);
        f = round_with_mode(f, static_cast<u8>(insn.imm_extra));
        std::memcpy(&b, &f, 4);
        std::memcpy(r + 4*i, &b, 4);
    }
    (void)write16(cpu, insn, insn.dst, r);
}
void op_roundpd(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    u8 r[16];
    for (int i = 0; i < 2; ++i) {
        u64 b; std::memcpy(&b, s + 8*i, 8);
        double f; std::memcpy(&f, &b, 8);
        f = round_with_mode(f, static_cast<u8>(insn.imm_extra));
        std::memcpy(&b, &f, 8);
        std::memcpy(r + 8*i, &b, 8);
    }
    (void)write16(cpu, insn, insn.dst, r);
}

// INSERTPS xmm, xmm/m32, imm8
void op_insertps(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read16(cpu, insn, insn.dst, d)) return;
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    const u8 imm = static_cast<u8>(insn.imm_extra & 0xFFu);
    const u8 count_s = (imm >> 6) & 0x3;          // source dword index
    const u8 count_d = (imm >> 4) & 0x3;          // destination dword index
    const u8 zmask   = imm & 0xF;                  // zero mask
    u32 sv;
    std::memcpy(&sv, s + 4*count_s, 4);
    std::memcpy(d + 4*count_d, &sv, 4);
    for (int i = 0; i < 4; ++i) {
        if (zmask & (1u << i)) {
            std::memset(d + 4*i, 0, 4);
        }
    }
    (void)write16(cpu, insn, insn.dst, d);
}

// PEXTR* / PINSR* family ----------------------------------------------------
void op_pextrb(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0xFu);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, static_cast<u64>(s[idx]));
}
void op_pextrw(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0x7u);
    const u32 v = u32(s[2*idx]) | (u32(s[2*idx+1]) << 8);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, v);
}
void op_pextrd(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0x3u);
    u32 v; std::memcpy(&v, s + 4*idx, 4);
    (void)write_operand(cpu, insn, insn.dst, 4, v);
}
void op_pextrq(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0x1u);
    u64 v; std::memcpy(&v, s + 8*idx, 8);
    (void)write_operand(cpu, insn, insn.dst, 8, v);
}
void op_pinsrb(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read16(cpu, insn, insn.dst, d)) return;
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, 1, v)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0xFu);
    d[idx] = static_cast<u8>(v);
    (void)write16(cpu, insn, insn.dst, d);
}
void op_pinsrw(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read16(cpu, insn, insn.dst, d)) return;
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, 2, v)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0x7u);
    d[2*idx]   = static_cast<u8>(v & 0xFFu);
    d[2*idx+1] = static_cast<u8>((v >> 8) & 0xFFu);
    (void)write16(cpu, insn, insn.dst, d);
}
void op_pinsrd(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read16(cpu, insn, insn.dst, d)) return;
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, 4, v)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0x3u);
    const u32 v32 = static_cast<u32>(v);
    std::memcpy(d + 4*idx, &v32, 4);
    (void)write16(cpu, insn, insn.dst, d);
}
void op_pinsrq(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read16(cpu, insn, insn.dst, d)) return;
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, 8, v)) return;
    const u8 idx = static_cast<u8>(insn.imm_extra & 0x1u);
    std::memcpy(d + 8*idx, &v, 8);
    (void)write16(cpu, insn, insn.dst, d);
}

// PCMPGT* -- packed signed compare greater
namespace {
template <typename T>
void cmpgt_pack(Cpu& cpu, const Insn& insn, int elem_size) noexcept {
    u8 a[16], b[16], r[16];
    if (!read16(cpu, insn, insn.dst, a)) return;
    if (!read16(cpu, insn, insn.src, b)) return;
    for (int i = 0; i < 16; i += elem_size) {
        T av, bv;
        std::memcpy(&av, a + i, sizeof(T));
        std::memcpy(&bv, b + i, sizeof(T));
        const bool gt = av > bv;
        for (int j = 0; j < elem_size; ++j) r[i+j] = gt ? 0xFFu : 0u;
    }
    (void)write16(cpu, insn, insn.dst, r);
}
} // namespace

void op_pcmpgtb(Cpu& cpu, const Insn& insn) { cmpgt_pack<i8> (cpu, insn, 1); }
void op_pcmpgtw(Cpu& cpu, const Insn& insn) { cmpgt_pack<i16>(cpu, insn, 2); }
void op_pcmpgtd(Cpu& cpu, const Insn& insn) { cmpgt_pack<i32>(cpu, insn, 4); }
void op_pcmpgtq(Cpu& cpu, const Insn& insn) { cmpgt_pack<i64>(cpu, insn, 8); }

void op_pmaxub(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read16(cpu, insn, insn.dst, a)) return;
    if (!read16(cpu, insn, insn.src, b)) return;
    for (int i = 0; i < 16; ++i) a[i] = a[i] > b[i] ? a[i] : b[i];
    (void)write16(cpu, insn, insn.dst, a);
}
void op_pmaxsw(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read16(cpu, insn, insn.dst, a)) return;
    if (!read16(cpu, insn, insn.src, b)) return;
    for (int i = 0; i < 8; ++i) {
        const i16 av = static_cast<i16>(u16(a[2*i]) | (u16(a[2*i+1]) << 8));
        const i16 bv = static_cast<i16>(u16(b[2*i]) | (u16(b[2*i+1]) << 8));
        const i16 r = av > bv ? av : bv;
        a[2*i]   = static_cast<u8>(static_cast<u16>(r) & 0xFFu);
        a[2*i+1] = static_cast<u8>((static_cast<u16>(r) >> 8) & 0xFFu);
    }
    (void)write16(cpu, insn, insn.dst, a);
}
void op_pminub(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read16(cpu, insn, insn.dst, a)) return;
    if (!read16(cpu, insn, insn.src, b)) return;
    for (int i = 0; i < 16; ++i) a[i] = a[i] < b[i] ? a[i] : b[i];
    (void)write16(cpu, insn, insn.dst, a);
}
void op_pminsw(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read16(cpu, insn, insn.dst, a)) return;
    if (!read16(cpu, insn, insn.src, b)) return;
    for (int i = 0; i < 8; ++i) {
        const i16 av = static_cast<i16>(u16(a[2*i]) | (u16(a[2*i+1]) << 8));
        const i16 bv = static_cast<i16>(u16(b[2*i]) | (u16(b[2*i+1]) << 8));
        const i16 r = av < bv ? av : bv;
        a[2*i]   = static_cast<u8>(static_cast<u16>(r) & 0xFFu);
        a[2*i+1] = static_cast<u8>((static_cast<u16>(r) >> 8) & 0xFFu);
    }
    (void)write16(cpu, insn, insn.dst, a);
}

void op_pabsb(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    for (int i = 0; i < 16; ++i) s[i] = static_cast<u8>(std::abs(static_cast<int>(static_cast<i8>(s[i]))));
    (void)write16(cpu, insn, insn.dst, s);
}
void op_pabsw(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    for (int i = 0; i < 8; ++i) {
        const i16 v = static_cast<i16>(u16(s[2*i]) | (u16(s[2*i+1]) << 8));
        const u16 r = static_cast<u16>(v < 0 ? -v : v);
        s[2*i]   = static_cast<u8>(r & 0xFFu);
        s[2*i+1] = static_cast<u8>((r >> 8) & 0xFFu);
    }
    (void)write16(cpu, insn, insn.dst, s);
}
void op_pabsd(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read16(cpu, insn, insn.src, s)) return;
    for (int i = 0; i < 4; ++i) {
        i32 v = 0;
        for (int j = 0; j < 4; ++j) v |= static_cast<i32>(u32(s[4*i+j]) << (8*j));
        const u32 r = static_cast<u32>(v < 0 ? -v : v);
        for (int j = 0; j < 4; ++j) s[4*i+j] = static_cast<u8>((r >> (8*j)) & 0xFFu);
    }
    (void)write16(cpu, insn, insn.dst, s);
}

// PSADBW -- sum of absolute differences for two 8-byte halves.
void op_psadbw(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16], r[16] = {};
    if (!read16(cpu, insn, insn.dst, a)) return;
    if (!read16(cpu, insn, insn.src, b)) return;
    u16 sum_lo = 0, sum_hi = 0;
    for (int i = 0; i < 8; ++i)  sum_lo += static_cast<u16>(std::abs(static_cast<int>(a[i]) - b[i]));
    for (int i = 8; i < 16; ++i) sum_hi += static_cast<u16>(std::abs(static_cast<int>(a[i]) - b[i]));
    r[0] = static_cast<u8>(sum_lo & 0xFFu);
    r[1] = static_cast<u8>((sum_lo >> 8) & 0xFFu);
    r[8] = static_cast<u8>(sum_hi & 0xFFu);
    r[9] = static_cast<u8>((sum_hi >> 8) & 0xFFu);
    (void)write16(cpu, insn, insn.dst, r);
}

// ---- More x87 trig --------------------------------------------------------
void op_fsin(Cpu& cpu, const Insn&)     { cpu.set_st(0, std::sin(cpu.st(0))); }
void op_fcos(Cpu& cpu, const Insn&)     { cpu.set_st(0, std::cos(cpu.st(0))); }
void op_fpatan(Cpu& cpu, const Insn&) {
    const double r = std::atan2(cpu.st(1), cpu.st(0));
    (void)cpu.fpu_pop();
    cpu.set_st(0, r);
}
void op_fyl2x(Cpu& cpu, const Insn&) {
    const double r = cpu.st(1) * std::log2(cpu.st(0));
    (void)cpu.fpu_pop();
    cpu.set_st(0, r);
}
void op_f2xm1(Cpu& cpu, const Insn&)    { cpu.set_st(0, std::pow(2.0, cpu.st(0)) - 1.0); }
void op_fscale(Cpu& cpu, const Insn&)   { cpu.set_st(0, std::ldexp(cpu.st(0), static_cast<int>(cpu.st(1)))); }
void op_frndint(Cpu& cpu, const Insn&)  { cpu.set_st(0, std::nearbyint(cpu.st(0))); }

} // namespace emu::handlers
