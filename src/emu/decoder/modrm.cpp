// ModRM + SIB + displacement decoding for x86-64.
//
// The r/m operand can be:
//   mod=11 -> register (full 64-bit GPR if REX.W; else 32-bit zero-extended).
//   mod=00,01,10 -> memory operand [base + index*scale + disp], possibly with
//                   RIP-relative or SIB encoding.
//
// We always lift to a 16-byte `Operand` (kind=Reg, Mem, or RipRelMem).

#include "decoder_internal.h"

namespace emu {

namespace {

u8 combine_reg(bool ext, u8 r3) noexcept {
    return static_cast<u8>((ext ? 8u : 0u) | (r3 & 7u));
}

// Parse a SIB byte (when ModRM.rm == 4 and mod != 3).
// Result: base reg (or NONE if "disp32 only" base=5 with mod=0), index reg
// (or NONE if index=4), scale.
struct Sib {
    u8 base;     // raw 4-bit base reg id (or reg::NONE)
    u8 index;    // raw 4-bit index reg id (or reg::NONE)
    u8 scale;    // 1/2/4/8
    bool needs_disp32_base;  // when base == 5 & mod == 0
};

Sib split_sib(u8 b, u8 mod, bool rex_x, bool rex_b) noexcept {
    const u8 scale_field = static_cast<u8>(b >> 6);
    const u8 index_field = static_cast<u8>((b >> 3) & 7);
    const u8 base_field  = static_cast<u8>(b & 7);

    Sib s{};
    s.scale = static_cast<u8>(1u << scale_field);

    // Index = 4 means "no index" (RSP can't be an index without REX.X).
    if (index_field == 4 && !rex_x) {
        s.index = reg::NONE;
    } else {
        s.index = combine_reg(rex_x, index_field);
    }

    if (base_field == 5 && mod == 0) {
        // Special: base = disp32 (no base reg).
        s.base = reg::NONE;
        s.needs_disp32_base = true;
    } else {
        s.base = combine_reg(rex_b, base_field);
        s.needs_disp32_base = false;
    }
    return s;
}

} // namespace

Operand reg_operand(const Prefixes& px, const ModRm& mrm, u8 op_size) noexcept {
    Operand op{};
    op.kind = OperandKind::Reg;
    // 8-bit register without REX prefix: indices 4..7 mean AH/CH/DH/BH
    // (high bytes of AX/CX/DX/BX), encoded internally as 16..19. With REX,
    // those indices stay as SPL/BPL/SIL/DIL (low bytes of RSP/RBP/RSI/RDI).
    if (op_size == 1 && !px.has_rex && mrm.reg >= 4 && mrm.reg <= 7) {
        op.reg = static_cast<u8>(16 + (mrm.reg - 4));
    } else {
        op.reg = combine_reg(px.rex_r(), mrm.reg);
    }
    return op;
}

Status decode_rm_operand(ByteSource& bs, const Prefixes& px, const ModRm& mrm,
                         u8 op_size, Operand& out) noexcept {
    out = {};

    if (mrm.mod == 3) {
        out.kind = OperandKind::Reg;
        if (op_size == 1 && !px.has_rex && mrm.rm >= 4 && mrm.rm <= 7) {
            out.reg = static_cast<u8>(16 + (mrm.rm - 4));
        } else {
            out.reg = combine_reg(px.rex_b(), mrm.rm);
        }
        return Status::Ok;
    }

    // Memory operand.
    out.seg = px.has_seg ? px.seg : Seg::DS;

    // Special case: mod=0, rm=5 -> RIP-relative (no SIB).
    if (mrm.mod == 0 && mrm.rm == 5) {
        u32 disp32 = 0;
        if (Status s = decoder::read_u32_ext(bs, disp32); fail(s)) return s;
        out.kind = OperandKind::RipRelMem;
        out.reg   = reg::NONE;
        out.index = reg::NONE;
        out.scale = 0;
        out.imm   = static_cast<i64>(static_cast<i32>(disp32));  // sign-extend
        (void)op_size;
        return Status::Ok;
    }

    // SIB present?
    if (mrm.rm == 4) {
        u8 sib_byte = 0;
        if (Status s = decoder::read_byte_ext(bs, sib_byte); fail(s)) return s;
        const Sib sib = split_sib(sib_byte, mrm.mod, px.rex_x(), px.rex_b());

        out.kind  = OperandKind::Mem;
        out.reg   = sib.base;
        out.index = sib.index;
        out.scale = sib.scale;

        i64 disp = 0;
        if (sib.needs_disp32_base) {
            u32 d32 = 0;
            if (Status s = decoder::read_u32_ext(bs, d32); fail(s)) return s;
            disp = static_cast<i64>(static_cast<i32>(d32));
        } else if (mrm.mod == 1) {
            u8 d8 = 0;
            if (Status s = decoder::read_byte_ext(bs, d8); fail(s)) return s;
            disp = static_cast<i64>(static_cast<i8>(d8));
        } else if (mrm.mod == 2) {
            u32 d32 = 0;
            if (Status s = decoder::read_u32_ext(bs, d32); fail(s)) return s;
            disp = static_cast<i64>(static_cast<i32>(d32));
        }
        out.imm = disp;
        (void)op_size;
        return Status::Ok;
    }

    // No SIB: base = REX.B | rm; disp per mod.
    out.kind  = OperandKind::Mem;
    out.reg   = combine_reg(px.rex_b(), mrm.rm);
    out.index = reg::NONE;
    out.scale = 0;

    i64 disp = 0;
    if (mrm.mod == 1) {
        u8 d8 = 0;
        if (Status s = decoder::read_byte_ext(bs, d8); fail(s)) return s;
        disp = static_cast<i64>(static_cast<i8>(d8));
    } else if (mrm.mod == 2) {
        u32 d32 = 0;
        if (Status s = decoder::read_u32_ext(bs, d32); fail(s)) return s;
        disp = static_cast<i64>(static_cast<i32>(d32));
    }
    out.imm = disp;
    (void)op_size;
    return Status::Ok;
}

Status read_imm_sx(ByteSource& bs, u8 imm_size, i64& out) noexcept {
    switch (imm_size) {
        case 1: {
            u8 b = 0;
            if (Status s = decoder::read_byte_ext(bs, b); fail(s)) return s;
            out = static_cast<i64>(static_cast<i8>(b));
            return Status::Ok;
        }
        case 2: {
            u16 v = 0;
            if (Status s = decoder::read_u16_ext(bs, v); fail(s)) return s;
            out = static_cast<i64>(static_cast<i16>(v));
            return Status::Ok;
        }
        case 4: {
            u32 v = 0;
            if (Status s = decoder::read_u32_ext(bs, v); fail(s)) return s;
            out = static_cast<i64>(static_cast<i32>(v));
            return Status::Ok;
        }
        case 8: {
            u64 v = 0;
            if (Status s = decoder::read_u64_ext(bs, v); fail(s)) return s;
            out = static_cast<i64>(v);
            return Status::Ok;
        }
        default:
            return Status::InvalidArgument;
    }
}

} // namespace emu
