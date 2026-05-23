// Phase 10 -- VEX-encoded 3-operand SSE, SHA-NI, SSE4.2 string compare.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cstring>

namespace emu::handlers {

namespace {

bool read_16_from(Cpu& cpu, const Insn& insn, const Operand& op, u8 out[16]) noexcept {
    if (op.kind == OperandKind::Xmm) { std::memcpy(out, cpu.xmm(op.reg), 16); return true; }
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "read16"); return false; }
    const GuestAddr addr = ea(cpu, insn, op);
    if (Status s = mp->read(addr, 16, out); fail(s)) { cpu.set_fault(FaultKind::PageFault, addr, s, "read16"); return false; }
    return true;
}

bool write_16_to(Cpu& cpu, const Insn& insn, const Operand& op, const u8 v[16]) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(cpu.xmm(op.reg), v, 16);
        if (insn.flags & INSN_FLAG_VEX) cpu.zero_ymm_upper(op.reg);   // VEX-128 zeroing
        return true;
    }
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "write16"); return false; }
    const GuestAddr addr = ea(cpu, insn, op);
    if (Status s = mp->write(addr, 16, v); fail(s)) { cpu.set_fault(kind_from_status(s), addr, s, "write16"); return false; }
    return true;
}

// VEX 3-operand binop driver. src1 from insn.dst.index (vvvv), src2 from insn.src,
// result to insn.dst.reg.
template <typename F>
void vex_binop_xmm(Cpu& cpu, const Insn& insn, F f) noexcept {
    u8 a[16], b[16], r[16];
    std::memcpy(a, cpu.xmm(insn.dst.index), 16);
    if (!read_16_from(cpu, insn, insn.src, b)) return;
    f(a, b, r);
    (void)write_16_to(cpu, insn, insn.dst, r);
}

} // namespace

// VPSHUFB ymm/xmm, vvvv, r/m  -- for each output byte, src[low4(b)] or 0 if b's MSB.
void op_vpshufb(Cpu& cpu, const Insn& insn) {
    vex_binop_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r){
        for (int i = 0; i < 16; ++i) {
            r[i] = (b[i] & 0x80) ? 0 : a[b[i] & 0x0F];
        }
    });
}

// VAES family -- delegates to the non-VEX AES handler logic with src1 from vvvv.
namespace {

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

inline u8 xt(u8 v) noexcept { return static_cast<u8>((v << 1) ^ ((v & 0x80) ? 0x1B : 0)); }

void shift_rows(u8 s[16]) noexcept {
    u8 t;
    t = s[1];  s[1] = s[5];  s[5] = s[9];  s[9] = s[13]; s[13] = t;
    t = s[2];  s[2] = s[10]; s[10] = t;
    t = s[6];  s[6] = s[14]; s[14] = t;
    t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
}
void inv_shift_rows(u8 s[16]) noexcept {
    u8 t;
    t = s[13]; s[13] = s[9]; s[9]  = s[5]; s[5]  = s[1]; s[1] = t;
    t = s[2];  s[2]  = s[10]; s[10] = t;
    t = s[6];  s[6]  = s[14]; s[14] = t;
    t = s[3];  s[3]  = s[7];  s[7]  = s[11]; s[11] = s[15]; s[15] = t;
}
void mix_columns(u8 s[16]) noexcept {
    for (int c = 0; c < 4; ++c) {
        const u8 a0 = s[4*c], a1 = s[4*c+1], a2 = s[4*c+2], a3 = s[4*c+3];
        const u8 x = a0 ^ a1 ^ a2 ^ a3;
        s[4*c]   = a0 ^ x ^ xt(a0 ^ a1);
        s[4*c+1] = a1 ^ x ^ xt(a1 ^ a2);
        s[4*c+2] = a2 ^ x ^ xt(a2 ^ a3);
        s[4*c+3] = a3 ^ x ^ xt(a3 ^ a0);
    }
}
void inv_mix_columns(u8 s[16]) noexcept {
    for (int c = 0; c < 4; ++c) {
        const u8 a0 = s[4*c], a1 = s[4*c+1], a2 = s[4*c+2], a3 = s[4*c+3];
        const u8 x = a0 ^ a1 ^ a2 ^ a3;
        const u8 y = xt(xt(x));
        const u8 t0 = a0 ^ y ^ xt(a0 ^ a1);
        const u8 t1 = a1 ^ y ^ xt(a1 ^ a2);
        const u8 t2 = a2 ^ y ^ xt(a2 ^ a3);
        const u8 t3 = a3 ^ y ^ xt(a3 ^ a0);
        s[4*c]   = t0 ^ xt(a0 ^ a2) ^ xt(xt(a0 ^ a2));
        s[4*c+1] = t1 ^ xt(a1 ^ a3) ^ xt(xt(a1 ^ a3));
        s[4*c+2] = t2 ^ xt(a0 ^ a2) ^ xt(xt(a0 ^ a2));
        s[4*c+3] = t3 ^ xt(a1 ^ a3) ^ xt(xt(a1 ^ a3));
    }
}

} // namespace

void op_vaesenc(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    std::memcpy(state, cpu.xmm(insn.dst.index), 16);     // src1 = vvvv
    if (!read_16_from(cpu, insn, insn.src, key)) return;
    shift_rows(state);
    for (int i = 0; i < 16; ++i) state[i] = SBOX[state[i]];
    mix_columns(state);
    for (int i = 0; i < 16; ++i) state[i] ^= key[i];
    (void)write_16_to(cpu, insn, insn.dst, state);
}
void op_vaesenclast(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    std::memcpy(state, cpu.xmm(insn.dst.index), 16);
    if (!read_16_from(cpu, insn, insn.src, key)) return;
    shift_rows(state);
    for (int i = 0; i < 16; ++i) state[i] = SBOX[state[i]];
    for (int i = 0; i < 16; ++i) state[i] ^= key[i];
    (void)write_16_to(cpu, insn, insn.dst, state);
}
void op_vaesdec(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    std::memcpy(state, cpu.xmm(insn.dst.index), 16);
    if (!read_16_from(cpu, insn, insn.src, key)) return;
    inv_shift_rows(state);
    for (int i = 0; i < 16; ++i) state[i] = INV_SBOX[state[i]];
    inv_mix_columns(state);
    for (int i = 0; i < 16; ++i) state[i] ^= key[i];
    (void)write_16_to(cpu, insn, insn.dst, state);
}
void op_vaesdeclast(Cpu& cpu, const Insn& insn) {
    u8 state[16], key[16];
    std::memcpy(state, cpu.xmm(insn.dst.index), 16);
    if (!read_16_from(cpu, insn, insn.src, key)) return;
    inv_shift_rows(state);
    for (int i = 0; i < 16; ++i) state[i] = INV_SBOX[state[i]];
    for (int i = 0; i < 16; ++i) state[i] ^= key[i];
    (void)write_16_to(cpu, insn, insn.dst, state);
}

void op_vpcmpgtq(Cpu& cpu, const Insn& insn) {
    vex_binop_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r){
        for (int i = 0; i < 2; ++i) {
            i64 av = 0, bv = 0;
            std::memcpy(&av, a + 8*i, 8);
            std::memcpy(&bv, b + 8*i, 8);
            const u64 mask = (av > bv) ? ~u64{0} : u64{0};
            std::memcpy(r + 8*i, &mask, 8);
        }
    });
}

void op_vpclmulqdq(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    std::memcpy(a, cpu.xmm(insn.dst.index), 16);
    if (!read_16_from(cpu, insn, insn.src, b)) return;
    const u8 imm = static_cast<u8>(insn.imm_extra & 0xFFu);
    u64 al, bl;
    std::memcpy(&al, a + ((imm & 0x1)  ? 8 : 0), 8);
    std::memcpy(&bl, b + ((imm & 0x10) ? 8 : 0), 8);
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
    (void)write_16_to(cpu, insn, insn.dst, r);
}

// ---- SHA-NI ---------------------------------------------------------------

namespace {

inline u32 read_lane32(const u8* p, int lane) noexcept {
    u32 v; std::memcpy(&v, p + 4*lane, 4); return v;
}
inline void write_lane32(u8* p, int lane, u32 v) noexcept {
    std::memcpy(p + 4*lane, &v, 4);
}
inline u32 rotl(u32 v, u8 n) noexcept { return (v << n) | (v >> (32 - n)); }
inline u32 rotr(u32 v, u8 n) noexcept { return (v >> n) | (v << (32 - n)); }

inline u32 sha1_f(u32 b, u32 c, u32 d, u8 t) noexcept {
    if      (t <= 19) return (b & c) | (~b & d);
    else if (t <= 39) return b ^ c ^ d;
    else if (t <= 59) return (b & c) | (b & d) | (c & d);
    else              return b ^ c ^ d;
}
inline u32 sha1_k(u8 t) noexcept {
    if      (t <= 19) return 0x5A827999u;
    else if (t <= 39) return 0x6ED9EBA1u;
    else if (t <= 59) return 0x8F1BBCDCu;
    else              return 0xCA62C1D6u;
}

} // namespace

// SHA1RNDS4 xmm, xmm/m128, imm8 -- 4 rounds of SHA-1 with imm8 selecting the round group.
void op_sha1rnds4(Cpu& cpu, const Insn& insn) {
    u8 dst[16], src[16];
    if (!read_16_from(cpu, insn, insn.dst, dst)) return;
    if (!read_16_from(cpu, insn, insn.src, src)) return;
    u32 a = read_lane32(dst, 3);
    u32 b = read_lane32(dst, 2);
    u32 c = read_lane32(dst, 1);
    u32 d = read_lane32(dst, 0);
    const u8 group = static_cast<u8>(insn.imm_extra & 0x3u);
    const u8 base_t = static_cast<u8>(group * 20);
    for (int i = 0; i < 4; ++i) {
        const u32 w = read_lane32(src, 3 - i);
        const u32 e_in = sha1_k(static_cast<u8>(base_t + i));   // round constant
        const u32 t = rotl(a, 5) + sha1_f(b, c, d, static_cast<u8>(base_t + i)) + e_in + w;
        d = c; c = rotl(b, 30); b = a; a = t;
    }
    write_lane32(dst, 3, a);
    write_lane32(dst, 2, b);
    write_lane32(dst, 1, c);
    write_lane32(dst, 0, d);
    (void)write_16_to(cpu, insn, insn.dst, dst);
}

void op_sha1nexte(Cpu& cpu, const Insn& insn) {
    u8 dst[16], src[16];
    if (!read_16_from(cpu, insn, insn.dst, dst)) return;
    if (!read_16_from(cpu, insn, insn.src, src)) return;
    const u32 e = read_lane32(dst, 3);
    u32 w3 = read_lane32(src, 3);
    w3 += rotl(e, 30);
    write_lane32(dst, 3, w3);
    write_lane32(dst, 2, read_lane32(src, 2));
    write_lane32(dst, 1, read_lane32(src, 1));
    write_lane32(dst, 0, read_lane32(src, 0));
    (void)write_16_to(cpu, insn, insn.dst, dst);
}

void op_sha1msg1(Cpu& cpu, const Insn& insn) {
    u8 dst[16], src[16];
    if (!read_16_from(cpu, insn, insn.dst, dst)) return;
    if (!read_16_from(cpu, insn, insn.src, src)) return;
    u32 w_2 = read_lane32(dst, 0);
    u32 w_1 = read_lane32(dst, 1);
    u32 w0  = read_lane32(dst, 2);
    u32 w1  = read_lane32(dst, 3);
    u32 w_3 = read_lane32(src, 3);
    write_lane32(dst, 3, w1 ^ w_3);
    write_lane32(dst, 2, w0 ^ w_2);
    write_lane32(dst, 1, w_1);
    write_lane32(dst, 0, w_2);
    (void)write_16_to(cpu, insn, insn.dst, dst);
}

void op_sha1msg2(Cpu& cpu, const Insn& insn) {
    u8 dst[16], src[16];
    if (!read_16_from(cpu, insn, insn.dst, dst)) return;
    if (!read_16_from(cpu, insn, insn.src, src)) return;
    u32 w13 = read_lane32(src, 2);
    u32 w14 = read_lane32(src, 1);
    u32 w15 = read_lane32(src, 0);
    u32 w16 = rotl(read_lane32(dst, 3) ^ w13, 1);
    u32 w17 = rotl(read_lane32(dst, 2) ^ w14, 1);
    u32 w18 = rotl(read_lane32(dst, 1) ^ w15, 1);
    u32 w19 = rotl(read_lane32(dst, 0) ^ w16, 1);
    write_lane32(dst, 3, w16);
    write_lane32(dst, 2, w17);
    write_lane32(dst, 1, w18);
    write_lane32(dst, 0, w19);
    (void)write_16_to(cpu, insn, insn.dst, dst);
}

namespace {
inline u32 ch(u32 x, u32 y, u32 z) noexcept   { return (x & y) ^ (~x & z); }
inline u32 maj(u32 x, u32 y, u32 z) noexcept  { return (x & y) ^ (x & z) ^ (y & z); }
inline u32 Sig0(u32 x) noexcept               { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline u32 Sig1(u32 x) noexcept               { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline u32 sig0(u32 x) noexcept               { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline u32 sig1(u32 x) noexcept               { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }
} // namespace

// SHA256RNDS2 xmm, xmm/m128 -- implicit xmm0 holds the round-constant pair.
void op_sha256rnds2(Cpu& cpu, const Insn& insn) {
    u8 dst[16], src[16], xmm0[16];
    if (!read_16_from(cpu, insn, insn.dst, dst)) return;
    if (!read_16_from(cpu, insn, insn.src, src)) return;
    std::memcpy(xmm0, cpu.xmm(0), 16);

    u32 a = read_lane32(src, 3);
    u32 b = read_lane32(src, 2);
    u32 c = read_lane32(dst, 3);
    u32 d = read_lane32(dst, 2);
    u32 e = read_lane32(src, 1);
    u32 f = read_lane32(src, 0);
    u32 g = read_lane32(dst, 1);
    u32 h = read_lane32(dst, 0);
    u32 w_lo = read_lane32(xmm0, 0);
    u32 w_hi = read_lane32(xmm0, 1);

    // Two rounds: WK0 then WK1
    {
        const u32 t1 = h + Sig1(e) + ch(e,f,g) + w_lo;
        const u32 t2 = Sig0(a) + maj(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    {
        const u32 t1 = h + Sig1(e) + ch(e,f,g) + w_hi;
        const u32 t2 = Sig0(a) + maj(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    write_lane32(dst, 3, a);
    write_lane32(dst, 2, b);
    write_lane32(dst, 1, e);
    write_lane32(dst, 0, f);
    (void)write_16_to(cpu, insn, insn.dst, dst);
}

void op_sha256msg1(Cpu& cpu, const Insn& insn) {
    u8 dst[16], src[16];
    if (!read_16_from(cpu, insn, insn.dst, dst)) return;
    if (!read_16_from(cpu, insn, insn.src, src)) return;
    for (int i = 0; i < 4; ++i) {
        const u32 w_lane = read_lane32(dst, i);
        const u32 w_src  = (i == 0) ? read_lane32(src, 3) : read_lane32(dst, i - 1);
        write_lane32(dst, i, w_lane + sig0(w_src));
    }
    (void)write_16_to(cpu, insn, insn.dst, dst);
}

void op_sha256msg2(Cpu& cpu, const Insn& insn) {
    u8 dst[16], src[16];
    if (!read_16_from(cpu, insn, insn.dst, dst)) return;
    if (!read_16_from(cpu, insn, insn.src, src)) return;
    u32 w14 = read_lane32(src, 2);
    u32 w15 = read_lane32(src, 3);
    u32 w16 = read_lane32(dst, 0) + sig1(w14);
    u32 w17 = read_lane32(dst, 1) + sig1(w15);
    u32 w18 = read_lane32(dst, 2) + sig1(w16);
    u32 w19 = read_lane32(dst, 3) + sig1(w17);
    write_lane32(dst, 0, w16);
    write_lane32(dst, 1, w17);
    write_lane32(dst, 2, w18);
    write_lane32(dst, 3, w19);
    (void)write_16_to(cpu, insn, insn.dst, dst);
}

// ---- PCMPESTRI / PCMPESTRM / PCMPISTRI / PCMPISTRM ------------------------
//
// We implement the most common mode (byte equality, equal-each, polarity=0,
// negative if no match). This covers `strchr`/`memchr`-style usage.

namespace {

void pcmpstr_core(Cpu& cpu, const Insn& insn, bool is_explicit, bool index_form) noexcept {
    u8 a[16], b[16];
    if (!read_16_from(cpu, insn, insn.dst, a)) return;
    if (!read_16_from(cpu, insn, insn.src, b)) return;
    const u8 ctrl = static_cast<u8>(insn.imm_extra & 0xFFu);
    const bool is_word = (ctrl & 0x1) != 0;
    const u8 lane_sz = is_word ? 2 : 1;
    const u8 lanes = is_word ? 8 : 16;

    u32 lenA = lanes, lenB = lanes;
    if (is_explicit) {
        // PCMPESTR: explicit lengths in EAX (a) and EDX (b), clamped to lanes.
        const u32 ea = static_cast<u32>(cpu.r32(reg::RAX));
        const u32 ed = static_cast<u32>(cpu.r32(reg::RDX));
        lenA = ea < lanes ? ea : lanes;
        lenB = ed < lanes ? ed : lanes;
    } else {
        // PCMPISTR: implicit lengths -- first zero lane defines the end.
        for (u32 i = 0; i < lanes; ++i) {
            const bool z = is_word
                ? (a[2*i] == 0 && a[2*i+1] == 0)
                : (a[i] == 0);
            if (z) { lenA = i; break; }
        }
        for (u32 i = 0; i < lanes; ++i) {
            const bool z = is_word
                ? (b[2*i] == 0 && b[2*i+1] == 0)
                : (b[i] == 0);
            if (z) { lenB = i; break; }
        }
    }

    // Build pairwise compare matrix (mode-0: equal-any, mode-1: ranges, mode-2:
    // equal-each, mode-3: equal-ordered). We implement equal-each (mode 2) as a
    // good default -- the most common usage.
    const u8 agg_mode = (ctrl >> 2) & 0x3u;
    u16 mask = 0;
    for (u32 i = 0; i < lanes; ++i) {
        if (i < lenA && i < lenB) {
            bool eq = is_word
                ? (a[2*i] == b[2*i] && a[2*i+1] == b[2*i+1])
                : (a[i] == b[i]);
            if (eq) mask |= (1u << i);
        }
    }
    if (agg_mode == 1) {
        // equal-any: for each byte of B, see if any byte of A matches.
        mask = 0;
        for (u32 j = 0; j < lenB; ++j) {
            for (u32 i = 0; i < lenA; ++i) {
                const bool eq = is_word
                    ? (a[2*i] == b[2*j] && a[2*i+1] == b[2*j+1])
                    : (a[i] == b[j]);
                if (eq) { mask |= (1u << j); break; }
            }
        }
    }

    // Polarity (bits 4-5): 00=none, 01=negate, 10=mask-by-validity, 11=...
    const u8 polarity = (ctrl >> 4) & 0x3u;
    if (polarity == 1) mask = static_cast<u16>(~mask);

    // Output: index form returns leading or trailing index in RCX; mask form
    // expands into XMM0.
    if (index_form) {
        const bool msb_first = (ctrl & 0x40) != 0;
        u32 idx = lanes;
        if (mask != 0) {
            if (msb_first) {
                for (int i = lanes - 1; i >= 0; --i) {
                    if (mask & (1u << i)) { idx = static_cast<u32>(i); break; }
                }
            } else {
                for (u32 i = 0; i < lanes; ++i) {
                    if (mask & (1u << i)) { idx = i; break; }
                }
            }
        }
        cpu.set_r64(reg::RCX, idx);
    } else {
        // Mask form: write byte mask (or expanded) into XMM0.
        u8 out[16] = {};
        const bool expand = (ctrl & 0x40) != 0;
        for (u32 i = 0; i < lanes; ++i) {
            const bool hit = (mask >> i) & 1u;
            if (expand) {
                for (u8 k = 0; k < lane_sz; ++k) out[i*lane_sz + k] = hit ? 0xFF : 0;
            } else {
                if (hit) out[i / 8] |= static_cast<u8>(1u << (i % 8));
            }
        }
        std::memcpy(cpu.xmm(0), out, 16);
    }

    // Flags: CF=mask!=0, ZF=lenB<lanes, SF=lenA<lanes, OF=mask[0], PF=AF=0
    cpu.force_sync_flags();
    cpu.set_cf(mask != 0);
    cpu.set_zf(lenB < lanes);
    cpu.set_sf(lenA < lanes);
    cpu.set_of((mask & 1u) != 0);
    cpu.set_pf(false); cpu.set_af(false);
}

} // namespace

void op_pcmpestri(Cpu& cpu, const Insn& insn) { pcmpstr_core(cpu, insn, /*explicit*/true,  /*index*/true);  }
void op_pcmpestrm(Cpu& cpu, const Insn& insn) { pcmpstr_core(cpu, insn, /*explicit*/true,  /*index*/false); }
void op_pcmpistri(Cpu& cpu, const Insn& insn) { pcmpstr_core(cpu, insn, /*explicit*/false, /*index*/true);  }
void op_pcmpistrm(Cpu& cpu, const Insn& insn) { pcmpstr_core(cpu, insn, /*explicit*/false, /*index*/false); }

} // namespace emu::handlers
