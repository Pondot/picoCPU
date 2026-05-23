// BMI1 / BMI2 -- VEX-encoded GPR bit-manipulation.
//
// All ops in this file read three operands: dst (a GPR via insn.dst.reg),
// vvvv (a second GPR via insn.dst.index), and r/m (via insn.src).
//
// Operand size is 32 or 64 depending on VEX.W; the decoder fills op_size.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <intrin.h>

namespace emu::handlers {

namespace {

u64 read_vvvv(const Cpu& cpu, const Insn& insn) noexcept {
    return cpu.r(insn.dst.index) & mask_for(insn.op_size);
}

bool read_src_u64(Cpu& cpu, const Insn& insn, u64& out) noexcept {
    return read_operand(cpu, insn, insn.src, insn.op_size, out);
}

} // namespace

// ANDN dst = ~vvvv & r/m
void op_andn(Cpu& cpu, const Insn& insn) {
    u64 b = 0;
    if (!read_src_u64(cpu, insn, b)) return;
    const u64 a = read_vvvv(cpu, insn);
    const u64 r = (~a) & b & mask_for(insn.op_size);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    cpu.stash_logic(insn.op_size, r);
}

// BEXTR dst, r/m, vvvv -- extract len bits starting at start, from r/m.
// vvvv low 8 = start, next 8 = len.
void op_bextr(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    const u64 ctl = read_vvvv(cpu, insn);
    const u8 start = static_cast<u8>(ctl & 0xFFu);
    const u8 len   = static_cast<u8>((ctl >> 8) & 0xFFu);
    const u8 width = static_cast<u8>(insn.op_size * 8);
    u64 r = 0;
    if (start < width) {
        const u8 eff_len = len > (width - start) ? (width - start) : len;
        const u64 mask = (eff_len >= 64) ? ~u64{0} : ((u64{1} << eff_len) - 1);
        r = (src >> start) & mask;
    }
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    cpu.force_sync_flags();
    cpu.set_zf(r == 0);
    cpu.set_cf(false); cpu.set_of(false); cpu.set_sf(false); cpu.set_pf(false); cpu.set_af(false);
}

// BLSI dst = r/m & -r/m  (isolate lowest set bit)
void op_blsi(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    src &= mask_for(insn.op_size);
    const u64 r = src & (0u - src);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    cpu.force_sync_flags();
    cpu.set_cf(src != 0);                            // CF = (src != 0)
    cpu.set_zf(r == 0);
    cpu.set_sf((r & sign_mask(insn.op_size)) != 0);
    cpu.set_of(false); cpu.set_pf(false); cpu.set_af(false);
}

// BLSR dst = r/m & (r/m - 1)  (clear lowest set bit)
void op_blsr(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    src &= mask_for(insn.op_size);
    const u64 r = src & (src - 1u);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    cpu.force_sync_flags();
    cpu.set_cf(src == 0);
    cpu.set_zf(r == 0);
    cpu.set_sf((r & sign_mask(insn.op_size)) != 0);
    cpu.set_of(false); cpu.set_pf(false); cpu.set_af(false);
}

// BLSMSK dst = r/m ^ (r/m - 1)
void op_blsmsk(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    src &= mask_for(insn.op_size);
    const u64 r = src ^ (src - 1u);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r & mask_for(insn.op_size));
    cpu.force_sync_flags();
    cpu.set_cf(src == 0);
    cpu.set_zf(false);
    cpu.set_sf((r & sign_mask(insn.op_size)) != 0);
    cpu.set_of(false); cpu.set_pf(false); cpu.set_af(false);
}

// BZHI dst = r/m & ((1 << (vvvv & 0xFF)) - 1)
void op_bzhi(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    const u8 n = static_cast<u8>(read_vvvv(cpu, insn) & 0xFFu);
    const u8 width = static_cast<u8>(insn.op_size * 8);
    u64 r;
    if (n >= width) {
        r = src & mask_for(insn.op_size);
    } else {
        const u64 mask = (u64{1} << n) - 1;
        r = src & mask;
    }
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    cpu.force_sync_flags();
    cpu.set_zf(r == 0);
    cpu.set_cf(n >= width);
    cpu.set_sf((r & sign_mask(insn.op_size)) != 0);
    cpu.set_of(false); cpu.set_pf(false); cpu.set_af(false);
}

// MULX dst, vvvv, r/m -- vvvv = low half, dst = high half, of vvvv * r/m,
// unsigned. Flags unchanged.
void op_mulx(Cpu& cpu, const Insn& insn) {
    u64 b = 0;
    if (!read_src_u64(cpu, insn, b)) return;
    // Implicit multiplier: rDX (per Intel SDM, MULX uses EDX/RDX as src1).
    const u64 a = cpu.r(reg::RDX) & mask_for(insn.op_size);
    u64 lo, hi;
    if (insn.op_size == 8) {
#if defined(_MSC_VER) && defined(_M_X64)
        unsigned __int64 h = 0;
        lo = _umul128(a, b, &h);
        hi = h;
#else
        lo = a * b;
        hi = 0;
#endif
    } else {
        const u64 p = a * b;
        lo = p & 0xFFFFFFFFull;
        hi = (p >> 32) & 0xFFFFFFFFull;
    }
    // Per SDM: vvvv gets low half, ModRM.reg gets high half.
    cpu.set_r64(insn.dst.index, lo);     // vvvv <- low
    cpu.set_r64(insn.dst.reg,   hi);     // ModRM.reg <- high
}

// PDEP -- parallel bits deposit.
void op_pdep(Cpu& cpu, const Insn& insn) {
    u64 mask = 0;
    if (!read_src_u64(cpu, insn, mask)) return;
    u64 src = read_vvvv(cpu, insn);
    u64 r = 0;
    u8 bit = 0;
    while (mask != 0) {
        const u64 lo = mask & (0u - mask);
        if ((src >> bit) & 1ull) r |= lo;
        mask &= mask - 1;
        ++bit;
    }
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
}

// PEXT -- parallel bits extract.
void op_pext(Cpu& cpu, const Insn& insn) {
    u64 mask = 0;
    if (!read_src_u64(cpu, insn, mask)) return;
    u64 src = read_vvvv(cpu, insn);
    u64 r = 0;
    u8 bit = 0;
    while (mask != 0) {
        const u64 lo = mask & (0u - mask);
        if (src & lo) r |= (u64{1} << bit);
        mask &= mask - 1;
        ++bit;
    }
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
}

// RORX dst = ROR(r/m, imm)  -- flags unchanged.
void op_rorx(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    src &= mask_for(insn.op_size);
    const u8 width = static_cast<u8>(insn.op_size * 8);
    const u8 n = static_cast<u8>(insn.imm_extra & (width - 1));
    const u64 r = ((src >> n) | (src << (width - n))) & mask_for(insn.op_size);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
}

// SARX / SHLX / SHRX -- variable shift; flags unchanged.
void op_sarx(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    const u8 n = static_cast<u8>(read_vvvv(cpu, insn) & shift_count_mask(insn.op_size));
    const i64 sign_ext = sign_extend(src, insn.op_size);
    const u64 r = static_cast<u64>(sign_ext >> n) & mask_for(insn.op_size);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
}
void op_shlx(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    const u8 n = static_cast<u8>(read_vvvv(cpu, insn) & shift_count_mask(insn.op_size));
    const u64 r = (src << n) & mask_for(insn.op_size);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
}
void op_shrx(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_src_u64(cpu, insn, src)) return;
    const u8 n = static_cast<u8>(read_vvvv(cpu, insn) & shift_count_mask(insn.op_size));
    const u64 r = (src & mask_for(insn.op_size)) >> n;
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
}

} // namespace emu::handlers
