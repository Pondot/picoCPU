// Miscellaneous handlers: LAHF/SAHF, PUSHF/POPF, INT, fences, prefetch,
// CLFLUSH, MOVBE.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace emu::handlers {

// LAHF: AH = SF:ZF:0:AF:0:PF:1:CF (low 8 bits of RFLAGS, with bit 1 forced to 1).
void op_lahf(Cpu& cpu, const Insn&) {
    cpu.force_sync_flags();
    const u8 ah = static_cast<u8>(cpu.rflags() & 0xD7) | 0x02;
    const u64 rax = cpu.r(reg::RAX);
    cpu.set_r64(reg::RAX, (rax & ~0xFF00ull) | (u64{ah} << 8));
}

// SAHF: low 8 bits of RFLAGS = AH (with the always-on bit preserved).
void op_sahf(Cpu& cpu, const Insn&) {
    const u8 ah = static_cast<u8>((cpu.r(reg::RAX) >> 8) & 0xFFu);
    cpu.force_sync_flags();
    u64 f = cpu.rflags() & ~u64{0xFF};
    f |= (ah & 0xD5) | 0x02;
    cpu.set_rflags(f);
}

// PUSHFQ: push RFLAGS (8 bytes).
void op_pushf(Cpu& cpu, const Insn&) {
    cpu.force_sync_flags();
    const u64 rflags = cpu.rflags();
    const u64 new_rsp = cpu.r(reg::RSP) - 8;
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, new_rsp, Status::ProviderFailure, "pushfq"); return; }
    if (Status s = mp->write(new_rsp, 8, &rflags); fail(s)) {
        cpu.set_fault(kind_from_status(s), new_rsp, s, "pushfq");
        return;
    }
    cpu.set_r64(reg::RSP, new_rsp);
}

// POPFQ: pop 8 bytes into RFLAGS.
void op_popf(Cpu& cpu, const Insn&) {
    const u64 rsp = cpu.r(reg::RSP);
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, rsp, Status::ProviderFailure, "popfq"); return; }
    u64 v = 0;
    if (Status s = mp->read(rsp, 8, &v); fail(s)) {
        cpu.set_fault(kind_from_status(s), rsp, s, "popfq");
        return;
    }
    cpu.set_rflags(v);
    cpu.set_r64(reg::RSP, rsp + 8);
}

// INT3 -- software breakpoint. Latches a #BP fault. Hookable via INSN(Int3).
void op_int3(Cpu& cpu, const Insn&) {
    cpu.set_fault(FaultKind::Breakpoint, cpu.rip(), Status::Ok, "INT3");
}

void op_int_n(Cpu& cpu, const Insn& insn) {
    // We don't service generic INT n; just record the vector and halt with #BP-ish.
    const u8 n = static_cast<u8>(insn.dst.imm & 0xFFu);
    char buf[24];
    std::snprintf(buf, sizeof(buf), "INT %u", static_cast<unsigned>(n));
    cpu.set_fault(FaultKind::Breakpoint, cpu.rip(), Status::Ok, "INT n");
}

// Fences / pause / cache control -- all observably no-ops in our model.
void op_fence(Cpu&, const Insn&) {}
void op_pause(Cpu&, const Insn&) {}
void op_clflush(Cpu&, const Insn&) {}
void op_prefetch(Cpu&, const Insn&) {}

// MOVBE: byte-reverse during memory ↔ register transfer.
void op_movbe(Cpu& cpu, const Insn& insn) {
    if (insn.dst.kind == OperandKind::Reg) {
        u64 v = 0;
        if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
        u64 r = 0;
        for (u8 i = 0; i < insn.op_size; ++i) {
            r |= ((v >> (8 * i)) & 0xFFull) << (8 * (insn.op_size - 1 - i));
        }
        (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    } else {
        // store form
        u64 v = 0;
        if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
        u64 r = 0;
        for (u8 i = 0; i < insn.op_size; ++i) {
            r |= ((v >> (8 * i)) & 0xFFull) << (8 * (insn.op_size - 1 - i));
        }
        (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    }
}

// --- Extra SSE -------------------------------------------------------------

namespace {

bool read_xmm(Cpu& cpu, const Insn& insn, const Operand& op, u8 out[16]) noexcept {
    if (op.kind == OperandKind::Xmm) { std::memcpy(out, cpu.xmm(op.reg), 16); return true; }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_read();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "xmm read"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->read(addr, 16, out); fail(s)) { cpu.set_fault(FaultKind::PageFault, addr, s, "xmm read"); return false; }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad src");
    return false;
}
bool write_xmm(Cpu& cpu, const Insn& insn, const Operand& op, const u8 v[16]) noexcept {
    if (op.kind == OperandKind::Xmm) { std::memcpy(cpu.xmm(op.reg), v, 16); return true; }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_write();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "xmm write"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->write(addr, 16, v); fail(s)) { cpu.set_fault(kind_from_status(s), addr, s, "xmm write"); return false; }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad dst");
    return false;
}

inline float  f32_from(u32 b) noexcept { float  f; std::memcpy(&f, &b, 4); return f; }
inline u32    bits_f32(float  f) noexcept { u32 b; std::memcpy(&b, &f, 4); return b; }

} // namespace

void op_rcpss(Cpu& cpu, const Insn& insn) {
    u8 dst[16]; if (!read_xmm(cpu, insn, insn.dst, dst)) return;
    u8 src[16]; if (!read_xmm(cpu, insn, insn.src, src)) return;
    u32 b; std::memcpy(&b, src, 4);
    const float r = 1.0f / f32_from(b);
    const u32 rb = bits_f32(r);
    std::memcpy(dst, &rb, 4);
    (void)write_xmm(cpu, insn, insn.dst, dst);
}
void op_rsqrtss(Cpu& cpu, const Insn& insn) {
    u8 dst[16]; if (!read_xmm(cpu, insn, insn.dst, dst)) return;
    u8 src[16]; if (!read_xmm(cpu, insn, insn.src, src)) return;
    u32 b; std::memcpy(&b, src, 4);
    const float v = f32_from(b);
    const float r = 1.0f / std::sqrt(v);
    const u32 rb = bits_f32(r);
    std::memcpy(dst, &rb, 4);
    (void)write_xmm(cpu, insn, insn.dst, dst);
}
void op_rcpps(Cpu& cpu, const Insn& insn) {
    u8 src[16]; if (!read_xmm(cpu, insn, insn.src, src)) return;
    u8 r[16];
    for (int i = 0; i < 4; ++i) {
        u32 b; std::memcpy(&b, src + 4*i, 4);
        const u32 rb = bits_f32(1.0f / f32_from(b));
        std::memcpy(r + 4*i, &rb, 4);
    }
    (void)write_xmm(cpu, insn, insn.dst, r);
}
void op_rsqrtps(Cpu& cpu, const Insn& insn) {
    u8 src[16]; if (!read_xmm(cpu, insn, insn.src, src)) return;
    u8 r[16];
    for (int i = 0; i < 4; ++i) {
        u32 b; std::memcpy(&b, src + 4*i, 4);
        const u32 rb = bits_f32(1.0f / std::sqrt(f32_from(b)));
        std::memcpy(r + 4*i, &rb, 4);
    }
    (void)write_xmm(cpu, insn, insn.dst, r);
}

// PMOVMSKB -- extract MSB of each byte in src (16 bytes) into a 16-bit mask in dst.
void op_pmovmskb(Cpu& cpu, const Insn& insn) {
    u8 src[16]; if (!read_xmm(cpu, insn, insn.src, src)) return;
    u32 mask = 0;
    for (int i = 0; i < 16; ++i) if (src[i] & 0x80) mask |= (1u << i);
    cpu.set_r64(insn.dst.reg, mask);
}
void op_movmskps(Cpu& cpu, const Insn& insn) {
    u8 src[16]; if (!read_xmm(cpu, insn, insn.src, src)) return;
    u32 mask = 0;
    for (int i = 0; i < 4; ++i) if (src[4*i + 3] & 0x80) mask |= (1u << i);
    cpu.set_r64(insn.dst.reg, mask);
}
void op_movmskpd(Cpu& cpu, const Insn& insn) {
    u8 src[16]; if (!read_xmm(cpu, insn, insn.src, src)) return;
    u32 mask = 0;
    if (src[7]  & 0x80) mask |= 1u;
    if (src[15] & 0x80) mask |= 2u;
    cpu.set_r64(insn.dst.reg, mask);
}

// SHUFPS / SHUFPD -- selector imm8 lives in insn.imm_extra.
void op_shufps(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read_xmm(cpu, insn, insn.dst, a)) return;
    if (!read_xmm(cpu, insn, insn.src, b)) return;
    const u8 imm = static_cast<u8>(insn.imm_extra & 0xFFu);
    u8 r[16];
    // Lanes 0,1 from a (low half of imm); lanes 2,3 from b (high half).
    auto pick = [&](const u8* x, u8 sel, int dst_lane) {
        std::memcpy(r + 4*dst_lane, x + 4*sel, 4);
    };
    pick(a, imm & 0x3,        0);
    pick(a, (imm >> 2) & 0x3, 1);
    pick(b, (imm >> 4) & 0x3, 2);
    pick(b, (imm >> 6) & 0x3, 3);
    (void)write_xmm(cpu, insn, insn.dst, r);
}
void op_shufpd(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read_xmm(cpu, insn, insn.dst, a)) return;
    if (!read_xmm(cpu, insn, insn.src, b)) return;
    const u8 imm = static_cast<u8>(insn.imm_extra & 0xFFu);
    u8 r[16];
    std::memcpy(r,     a + ((imm & 1) ? 8 : 0), 8);
    std::memcpy(r + 8, b + ((imm & 2) ? 8 : 0), 8);
    (void)write_xmm(cpu, insn, insn.dst, r);
}

// PSHUFB -- for each byte in dst, pick src[low4bits of dst-byte] or 0 if MSB set.
void op_pshufb(Cpu& cpu, const Insn& insn) {
    u8 d[16], s[16];
    if (!read_xmm(cpu, insn, insn.dst, d)) return;
    if (!read_xmm(cpu, insn, insn.src, s)) return;
    u8 r[16];
    for (int i = 0; i < 16; ++i) {
        if (s[i] & 0x80) r[i] = 0;
        else             r[i] = d[s[i] & 0x0F];
    }
    (void)write_xmm(cpu, insn, insn.dst, r);
}

void op_pshufhw(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read_xmm(cpu, insn, insn.src, s)) return;
    const u8 imm = static_cast<u8>(insn.imm_extra & 0xFFu);
    u8 r[16];
    std::memcpy(r, s, 8);                // low qword unchanged
    auto pick16 = [&](u8 sel, int lane) {
        std::memcpy(r + 8 + 2*lane, s + 8 + 2*sel, 2);
    };
    pick16(imm & 0x3,        0);
    pick16((imm >> 2) & 0x3, 1);
    pick16((imm >> 4) & 0x3, 2);
    pick16((imm >> 6) & 0x3, 3);
    (void)write_xmm(cpu, insn, insn.dst, r);
}
void op_pshuflw(Cpu& cpu, const Insn& insn) {
    u8 s[16]; if (!read_xmm(cpu, insn, insn.src, s)) return;
    const u8 imm = static_cast<u8>(insn.imm_extra & 0xFFu);
    u8 r[16];
    std::memcpy(r + 8, s + 8, 8);        // high qword unchanged
    auto pick16 = [&](u8 sel, int lane) {
        std::memcpy(r + 2*lane, s + 2*sel, 2);
    };
    pick16(imm & 0x3,        0);
    pick16((imm >> 2) & 0x3, 1);
    pick16((imm >> 4) & 0x3, 2);
    pick16((imm >> 6) & 0x3, 3);
    (void)write_xmm(cpu, insn, insn.dst, r);
}

// PMULLW / PMULHW / PMULHUW: 8x packed 16-bit multiply.
namespace {
template <typename Combine>
void packed_mul16(Cpu& cpu, const Insn& insn, Combine f) noexcept {
    u8 a[16], b[16];
    if (!read_xmm(cpu, insn, insn.dst, a)) return;
    if (!read_xmm(cpu, insn, insn.src, b)) return;
    for (int i = 0; i < 8; ++i) {
        const u16 av = u16(a[2*i]) | (u16(a[2*i+1]) << 8);
        const u16 bv = u16(b[2*i]) | (u16(b[2*i+1]) << 8);
        const u16 r  = f(av, bv);
        a[2*i]   = static_cast<u8>(r & 0xFFu);
        a[2*i+1] = static_cast<u8>((r >> 8) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, a);
}
} // namespace

void op_pmullw(Cpu& cpu, const Insn& insn) {
    packed_mul16(cpu, insn, [](u16 a, u16 b){
        const u32 p = u32(a) * u32(b);
        return static_cast<u16>(p & 0xFFFFu);
    });
}
void op_pmulhw(Cpu& cpu, const Insn& insn) {
    packed_mul16(cpu, insn, [](u16 a, u16 b){
        const i32 p = static_cast<i32>(static_cast<i16>(a)) * static_cast<i32>(static_cast<i16>(b));
        return static_cast<u16>((static_cast<u32>(p) >> 16) & 0xFFFFu);
    });
}
void op_pmulhuw(Cpu& cpu, const Insn& insn) {
    packed_mul16(cpu, insn, [](u16 a, u16 b){
        const u32 p = u32(a) * u32(b);
        return static_cast<u16>((p >> 16) & 0xFFFFu);
    });
}

// PMADDWD: 4× 32-bit (a_2i*b_2i + a_2i+1*b_2i+1).
void op_pmaddwd(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16];
    if (!read_xmm(cpu, insn, insn.dst, a)) return;
    if (!read_xmm(cpu, insn, insn.src, b)) return;
    u8 r[16];
    for (int i = 0; i < 4; ++i) {
        const i16 a0 = static_cast<i16>(u16(a[4*i])   | (u16(a[4*i+1]) << 8));
        const i16 a1 = static_cast<i16>(u16(a[4*i+2]) | (u16(a[4*i+3]) << 8));
        const i16 b0 = static_cast<i16>(u16(b[4*i])   | (u16(b[4*i+1]) << 8));
        const i16 b1 = static_cast<i16>(u16(b[4*i+2]) | (u16(b[4*i+3]) << 8));
        const i32 d  = static_cast<i32>(a0) * static_cast<i32>(b0)
                     + static_cast<i32>(a1) * static_cast<i32>(b1);
        const u32 u = static_cast<u32>(d);
        for (int j = 0; j < 4; ++j) r[4*i+j] = static_cast<u8>((u >> (8*j)) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, r);
}

// PSLL/PSRL/PSRA -- packed shift by count (imm or in src low 64 bits).
namespace {
u8 shift_count_from(Cpu& cpu, const Insn& insn) noexcept {
    if (insn.src.kind == OperandKind::Imm) {
        return static_cast<u8>(insn.imm_extra & 0xFFu);
    }
    u8 s[16] = {};
    if (insn.src.kind == OperandKind::Xmm) std::memcpy(s, cpu.xmm(insn.src.reg), 16);
    return s[0];
}
} // namespace

void op_psllw(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 8; ++i) {
        u16 v = u16(d[2*i]) | (u16(d[2*i+1]) << 8);
        v = (c >= 16) ? 0 : static_cast<u16>(v << c);
        d[2*i]   = static_cast<u8>(v & 0xFFu);
        d[2*i+1] = static_cast<u8>((v >> 8) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}
void op_pslld(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 4; ++i) {
        u32 v = 0; for (int j = 0; j < 4; ++j) v |= u32(d[4*i+j]) << (8*j);
        v = (c >= 32) ? 0 : (v << c);
        for (int j = 0; j < 4; ++j) d[4*i+j] = static_cast<u8>((v >> (8*j)) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}
void op_psllq(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 2; ++i) {
        u64 v = 0; for (int j = 0; j < 8; ++j) v |= u64(d[8*i+j]) << (8*j);
        v = (c >= 64) ? 0 : (v << c);
        for (int j = 0; j < 8; ++j) d[8*i+j] = static_cast<u8>((v >> (8*j)) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}

void op_psrlw(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 8; ++i) {
        u16 v = u16(d[2*i]) | (u16(d[2*i+1]) << 8);
        v = (c >= 16) ? 0 : static_cast<u16>(v >> c);
        d[2*i]   = static_cast<u8>(v & 0xFFu);
        d[2*i+1] = static_cast<u8>((v >> 8) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}
void op_psrld(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 4; ++i) {
        u32 v = 0; for (int j = 0; j < 4; ++j) v |= u32(d[4*i+j]) << (8*j);
        v = (c >= 32) ? 0 : (v >> c);
        for (int j = 0; j < 4; ++j) d[4*i+j] = static_cast<u8>((v >> (8*j)) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}
void op_psrlq(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 2; ++i) {
        u64 v = 0; for (int j = 0; j < 8; ++j) v |= u64(d[8*i+j]) << (8*j);
        v = (c >= 64) ? 0 : (v >> c);
        for (int j = 0; j < 8; ++j) d[8*i+j] = static_cast<u8>((v >> (8*j)) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}

void op_psraw(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 8; ++i) {
        const i16 v = static_cast<i16>(u16(d[2*i]) | (u16(d[2*i+1]) << 8));
        const i16 r = (c >= 16) ? static_cast<i16>(v >> 15) : static_cast<i16>(v >> c);
        d[2*i]   = static_cast<u8>(static_cast<u16>(r) & 0xFFu);
        d[2*i+1] = static_cast<u8>((static_cast<u16>(r) >> 8) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}
void op_psrad(Cpu& cpu, const Insn& insn) {
    u8 d[16]; if (!read_xmm(cpu, insn, insn.dst, d)) return;
    const u8 c = shift_count_from(cpu, insn);
    for (int i = 0; i < 4; ++i) {
        i32 v = 0; for (int j = 0; j < 4; ++j) v |= static_cast<i32>(u32(d[4*i+j]) << (8*j));
        const i32 r = (c >= 32) ? (v >> 31) : (v >> c);
        for (int j = 0; j < 4; ++j) d[4*i+j] = static_cast<u8>((static_cast<u32>(r) >> (8*j)) & 0xFFu);
    }
    (void)write_xmm(cpu, insn, insn.dst, d);
}

// UNPCKL / UNPCKH -- interleave low/high halves of two xmm regs.
namespace {
void unpck(Cpu& cpu, const Insn& insn, int elem_size, bool hi) noexcept {
    u8 a[16], b[16], r[16];
    if (!read_xmm(cpu, insn, insn.dst, a)) return;
    if (!read_xmm(cpu, insn, insn.src, b)) return;
    const int start = hi ? 8 : 0;
    int out_off = 0;
    for (int pair = 0; pair < 8 / elem_size && out_off + 2*elem_size <= 16; ++pair) {
        std::memcpy(r + out_off, a + start + pair * elem_size, elem_size); out_off += elem_size;
        std::memcpy(r + out_off, b + start + pair * elem_size, elem_size); out_off += elem_size;
    }
    (void)write_xmm(cpu, insn, insn.dst, r);
}
} // namespace

void op_punpcklbw (Cpu& cpu, const Insn& insn) { unpck(cpu, insn, 1, false); }
void op_punpcklwd (Cpu& cpu, const Insn& insn) { unpck(cpu, insn, 2, false); }
void op_punpckldq (Cpu& cpu, const Insn& insn) { unpck(cpu, insn, 4, false); }
void op_punpcklqdq(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16], r[16];
    if (!read_xmm(cpu, insn, insn.dst, a)) return;
    if (!read_xmm(cpu, insn, insn.src, b)) return;
    std::memcpy(r,     a, 8);
    std::memcpy(r + 8, b, 8);
    (void)write_xmm(cpu, insn, insn.dst, r);
}
void op_punpckhbw (Cpu& cpu, const Insn& insn) { unpck(cpu, insn, 1, true); }
void op_punpckhwd (Cpu& cpu, const Insn& insn) { unpck(cpu, insn, 2, true); }
void op_punpckhdq (Cpu& cpu, const Insn& insn) { unpck(cpu, insn, 4, true); }
void op_punpckhqdq(Cpu& cpu, const Insn& insn) {
    u8 a[16], b[16], r[16];
    if (!read_xmm(cpu, insn, insn.dst, a)) return;
    if (!read_xmm(cpu, insn, insn.src, b)) return;
    std::memcpy(r,     a + 8, 8);
    std::memcpy(r + 8, b + 8, 8);
    (void)write_xmm(cpu, insn, insn.dst, r);
}

} // namespace emu::handlers
