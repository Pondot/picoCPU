// Shift / rotate handlers: SHL / SHR / SAR / ROL / ROR.
//
// RCL/RCR/SHLD/SHRD land in Phase 4.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

namespace emu::handlers {

namespace {

// Read shift count (op.src is either Imm or RCX). Masked per operand size.
bool read_count(Cpu& cpu, const Insn& insn, u8& count) noexcept {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, 1, v)) return false;
    count = static_cast<u8>(v & shift_count_mask(insn.op_size));
    return true;
}

} // namespace

void op_shl(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    u8 c = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_count(cpu, insn, c)) return;
    const u64 m = mask_for(insn.op_size);
    const u8  w = static_cast<u8>(insn.op_size * 8);
    const u64 r = (c >= w) ? 0 : ((a << c) & m);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_shl(insn.op_size, a, c, r);
}

void op_shr(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    u8 c = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_count(cpu, insn, c)) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u64 r = (c >= w) ? 0 : ((a & mask_for(insn.op_size)) >> c);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_shr(insn.op_size, a, c, r);
}

void op_sar(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    u8 c = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_count(cpu, insn, c)) return;
    const i64 sa = sign_extend(a, insn.op_size);
    const u8 w = static_cast<u8>(insn.op_size * 8);
    i64 sr;
    if (c >= w) sr = sa < 0 ? -1 : 0;
    else        sr = sa >> c;
    const u64 r = static_cast<u64>(sr) & mask_for(insn.op_size);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_sar(insn.op_size, a, c, r);
}

void op_rol(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    u8 c = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_count(cpu, insn, c)) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u8 mod = static_cast<u8>(c & (w - 1u));   // x86 spec: count modulo width for ROL/ROR
    const u64 m = mask_for(insn.op_size);
    const u64 av = a & m;
    const u64 r = mod == 0 ? av : (((av << mod) | (av >> (w - mod))) & m);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_rol(insn.op_size, a, c, r);
}

void op_ror(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    u8 c = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_count(cpu, insn, c)) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u8 mod = static_cast<u8>(c & (w - 1u));
    const u64 m = mask_for(insn.op_size);
    const u64 av = a & m;
    const u64 r = mod == 0 ? av : (((av >> mod) | (av << (w - mod))) & m);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_ror(insn.op_size, a, c, r);
}

// RCL/RCR are rotates through the carry flag. The effective bit-width is
// (op_size*8 + 1) because CF participates as an extra bit. We pre-sync flags
// so we have the *current* CF, do the rotate, then write back CF directly --
// no lazy stash, since RCL/RCR's flag effect is too tangled for the cc_op
// model and they're rare enough that eager-only is fine.
void op_rcl(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    u8  c = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_count(cpu, insn, c)) return;
    if (c == 0) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u8 mod = static_cast<u8>(c % (w + 1));
    if (mod == 0) return;
    cpu.force_sync_flags();
    u64 val = a & mask_for(insn.op_size);
    bool cf  = cpu.cf();
    for (u8 i = 0; i < mod; ++i) {
        const bool new_cf = (val & sign_mask(insn.op_size)) != 0;
        val = (val << 1) | (cf ? 1ull : 0ull);
        val &= mask_for(insn.op_size);
        cf = new_cf;
    }
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, val)) return;
    cpu.set_cf(cf);
    if (c == 1) {
        const bool sign = (val & sign_mask(insn.op_size)) != 0;
        cpu.set_of(sign != cf);
    }
}

void op_rcr(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    u8  c = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_count(cpu, insn, c)) return;
    if (c == 0) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u8 mod = static_cast<u8>(c % (w + 1));
    if (mod == 0) return;
    cpu.force_sync_flags();
    u64 val = a & mask_for(insn.op_size);
    bool cf  = cpu.cf();
    const u64 sb = sign_mask(insn.op_size);
    if (c == 1) {
        // OF for RCR by 1 is computed from MSB before the rotate.
        cpu.set_of(((val & sb) != 0) != cf);
    }
    for (u8 i = 0; i < mod; ++i) {
        const bool new_cf = (val & 1ull) != 0;
        val = (val >> 1) | (cf ? sb : 0ull);
        cf = new_cf;
    }
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, val)) return;
    cpu.set_cf(cf);
}

// SHLD r/m, r, count   ->   r/m = (r/m << count) | (r >> (W - count))
// SHRD r/m, r, count   ->   r/m = (r/m >> count) | (r << (W - count))
//
// The decoder packs the secondary register's id into `dst.index` (a reuse of
// the otherwise-unused-for-Reg index field) and the count into `imm_extra`
// for the imm8 form or `src.reg = RCX` for the CL form.

void op_shld(Cpu& cpu, const Insn& insn) {
    u64 dst = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, dst)) return;
    const u64 src = cpu.r(insn.dst.index) & mask_for(insn.op_size);
    u8 count = 0;
    if (insn.src.kind == OperandKind::Reg) {
        count = static_cast<u8>(cpu.r8(reg::RCX) & shift_count_mask(insn.op_size));
    } else {
        count = static_cast<u8>(static_cast<u64>(insn.imm_extra) & shift_count_mask(insn.op_size));
    }
    if (count == 0) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    if (count >= w) {
        // SDM: undefined for count >= width. Match what real hardware does:
        // result is unpredictable. We choose all-zero so emulator output is
        // deterministic.
        if (!write_operand(cpu, insn, insn.dst, insn.op_size, 0)) return;
        cpu.stash_logic(insn.op_size, 0);
        return;
    }
    const u64 r = ((dst << count) | (src >> (w - count))) & mask_for(insn.op_size);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    // CF = last bit shifted out (bit at position W-count in dst).
    cpu.stash_shl(insn.op_size, dst, count, r);
}

void op_shrd(Cpu& cpu, const Insn& insn) {
    u64 dst = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, dst)) return;
    const u64 src = cpu.r(insn.dst.index) & mask_for(insn.op_size);
    u8 count = 0;
    if (insn.src.kind == OperandKind::Reg) {
        count = static_cast<u8>(cpu.r8(reg::RCX) & shift_count_mask(insn.op_size));
    } else {
        count = static_cast<u8>(static_cast<u64>(insn.imm_extra) & shift_count_mask(insn.op_size));
    }
    if (count == 0) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    if (count >= w) {
        if (!write_operand(cpu, insn, insn.dst, insn.op_size, 0)) return;
        cpu.stash_logic(insn.op_size, 0);
        return;
    }
    const u64 r = ((dst >> count) | (src << (w - count))) & mask_for(insn.op_size);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_shr(insn.op_size, dst, count, r);
}

} // namespace emu::handlers
