// x86-64 decoder -- entry points and the prefix + REX + opcode dispatcher.
//
// We grow this file as we add coverage. For Phase 1 we support a useful
// subset of the GPR ISA: all 1-byte primary opcodes used by our mixer, plus
// the 2-byte 0F AF (IMUL r, r/m).
//
// The decoder is *not* on the hot path of emulation -- it runs once per
// distinct PC (the block cache memoizes the output). We optimize for clarity.

#include "emu/decoder.h"

#include "emu/cpu.h"
#include "emu/logger.h"

#include "decoder_internal.h"

#include <cstring>

namespace emu {

namespace {

// `ByteSource` is declared in decoder_internal.h; we just supply
// concrete contexts + fetch functions here.

struct BufCtx {
    const u8* data;
    usize     size;
};

Status buf_fetch(void* ctx, u8 off, u8& out) {
    auto* c = static_cast<BufCtx*>(ctx);
    if (off >= c->size) return Status::TruncatedInstruction;
    out = c->data[off];
    return Status::Ok;
}

struct MemCtx {
    MemoryProvider* mem;
    GuestAddr       rip;
};

Status mem_fetch(void* ctx, u8 off, u8& out) {
    auto* c = static_cast<MemCtx*>(ctx);
    u8 b = 0;
    if (Status s = c->mem->read(c->rip + off, 1, &b); fail(s)) return s;
    out = b;
    return Status::Ok;
}

Status read_byte(ByteSource& bs, u8& out) noexcept {
    if (Status s = bs.fetch(bs.ctx, bs.consumed, out); fail(s)) return s;
    bs.consumed = static_cast<u8>(bs.consumed + 1);
    return Status::Ok;
}

Status peek_byte(ByteSource& bs, u8 off_from_cur, u8& out) noexcept {
    return bs.fetch(bs.ctx, static_cast<u8>(bs.consumed + off_from_cur), out);
}

Status read_u16(ByteSource& bs, u16& out) noexcept {
    u8 lo = 0, hi = 0;
    if (Status s = read_byte(bs, lo); fail(s)) return s;
    if (Status s = read_byte(bs, hi); fail(s)) return s;
    out = static_cast<u16>(lo | (u16(hi) << 8));
    return Status::Ok;
}

Status read_u32(ByteSource& bs, u32& out) noexcept {
    u8 b[4]{};
    for (auto& v : b) {
        if (Status s = read_byte(bs, v); fail(s)) return s;
    }
    out = u32(b[0]) | (u32(b[1]) << 8) | (u32(b[2]) << 16) | (u32(b[3]) << 24);
    return Status::Ok;
}

Status read_u64(ByteSource& bs, u64& out) noexcept {
    u8 b[8]{};
    for (auto& v : b) {
        if (Status s = read_byte(bs, v); fail(s)) return s;
    }
    out = 0;
    for (int i = 7; i >= 0; --i) out = (out << 8) | b[i];
    return Status::Ok;
}

} // namespace

// ---- Forward decls for the table-driven opcode handlers --------------------
namespace decoder {

// Implemented in decoder_opcodes.cpp.
Status decode_after_prefixes(ByteSource& bs, Prefixes& px, GuestAddr rip, Insn& out) noexcept;

} // namespace decoder

namespace {

// Apply VEX `pp` bits to the legacy-prefix slots so downstream decoder code
// can stay prefix-agnostic.
void apply_vex_pp(Prefixes& px, u8 pp) noexcept {
    switch (pp & 0x3) {
        case 0: break;
        case 1: px.opsize_pfx = true; break;
        case 2: px.rep        = true; break;
        case 3: px.repne      = true; break;
    }
}

// Scan legacy prefixes + REX/VEX. After this, `bs` is positioned at the opcode.
Status scan_prefixes(ByteSource& bs, Prefixes& px) noexcept {
    px = {};
    for (;;) {
        u8 b = 0;
        if (Status s = peek_byte(bs, 0, b); fail(s)) return s;
        switch (b) {
            case 0x66: px.opsize_pfx   = true; (void)read_byte(bs, b); continue;
            case 0x67: px.addrsize_pfx = true; (void)read_byte(bs, b); continue;
            case 0xF0: px.lock         = true; (void)read_byte(bs, b); continue;
            case 0xF2: px.repne        = true; (void)read_byte(bs, b); continue;
            case 0xF3: px.rep          = true; (void)read_byte(bs, b); continue;
            case 0x26: px.seg = Seg::ES; px.has_seg = true; (void)read_byte(bs, b); continue;
            case 0x2E: px.seg = Seg::CS; px.has_seg = true; (void)read_byte(bs, b); continue;
            case 0x36: px.seg = Seg::SS; px.has_seg = true; (void)read_byte(bs, b); continue;
            case 0x3E: px.seg = Seg::DS; px.has_seg = true; (void)read_byte(bs, b); continue;
            case 0x64: px.seg = Seg::FS; px.has_seg = true; (void)read_byte(bs, b); continue;
            case 0x65: px.seg = Seg::GS; px.has_seg = true; (void)read_byte(bs, b); continue;
            default: break;
        }
        // VEX 2-byte: C5
        if (b == 0xC5) {
            (void)read_byte(bs, b);
            u8 v = 0;
            if (Status s = read_byte(bs, v); fail(s)) return s;
            px.is_vex   = true;
            px.vex_map  = 1;                                 // implicit 0F
            px.vex_r_   = (v & 0x80) == 0;                   // inverted
            px.vex_x_   = false;                              // no X in 2-byte VEX
            px.vex_b_   = false;
            px.vex_vvvv = static_cast<u8>((~(v >> 3)) & 0xF);
            px.vex_l    = (v & 0x04) != 0;
            apply_vex_pp(px, v & 0x3);
            return Status::Ok;
        }
        // EVEX 4-byte: 62 (AVX-512)
        if (b == 0x62) {
            (void)read_byte(bs, b);
            u8 p1 = 0, p2 = 0, p3 = 0;
            if (Status s = read_byte(bs, p1); fail(s)) return s;
            if (Status s = read_byte(bs, p2); fail(s)) return s;
            if (Status s = read_byte(bs, p3); fail(s)) return s;
            // EVEX validation: bit 2 of p1 must be 0, bit 3 of p2 must be 1.
            // Modern parsers accept both for forward compat; we do the same.
            px.is_vex  = true;
            px.is_evex = true;
            px.vex_r_   = (p1 & 0x80) == 0;
            px.vex_x_   = (p1 & 0x40) == 0;
            px.vex_b_   = (p1 & 0x20) == 0;
            px.evex_r_  = (p1 & 0x10) == 0;
            px.vex_map  = static_cast<u8>(p1 & 0x07);
            px.vex_w    = (p2 & 0x80) != 0;
            px.vex_vvvv = static_cast<u8>((~(p2 >> 3)) & 0x0F);
            apply_vex_pp(px, p2 & 0x3);
            px.evex_z   = (p3 & 0x80) != 0;
            px.evex_l2  = (p3 & 0x40) != 0;
            px.vex_l    = (p3 & 0x20) != 0;
            px.evex_b   = (p3 & 0x10) != 0;
            px.evex_v_  = (p3 & 0x08) == 0;     // V' inverted
            px.evex_aaa = static_cast<u8>(p3 & 0x07);
            // Extend vvvv to 5 bits (V' is the high bit when used).
            if (!px.evex_v_) px.vex_vvvv = static_cast<u8>(px.vex_vvvv | 0x10);
            return Status::Ok;
        }

        // VEX 3-byte: C4
        if (b == 0xC4) {
            (void)read_byte(bs, b);
            u8 b1 = 0, b2 = 0;
            if (Status s = read_byte(bs, b1); fail(s)) return s;
            if (Status s = read_byte(bs, b2); fail(s)) return s;
            px.is_vex   = true;
            px.vex_r_   = (b1 & 0x80) == 0;
            px.vex_x_   = (b1 & 0x40) == 0;
            px.vex_b_   = (b1 & 0x20) == 0;
            px.vex_map  = static_cast<u8>(b1 & 0x1F);
            px.vex_w    = (b2 & 0x80) != 0;
            px.vex_vvvv = static_cast<u8>((~(b2 >> 3)) & 0xF);
            px.vex_l    = (b2 & 0x04) != 0;
            apply_vex_pp(px, b2 & 0x3);
            return Status::Ok;
        }
        // REX must be immediately before the opcode (legacy path).
        if (b >= 0x40 && b <= 0x4F) {
            px.rex     = b;
            px.has_rex = true;
            (void)read_byte(bs, b);
        }
        break;
    }
    return Status::Ok;
}

Status decode_with_source(ByteSource& bs, GuestAddr rip, Insn& out) noexcept {
    Prefixes px{};
    if (Status s = scan_prefixes(bs, px); fail(s)) return s;

    out = {};
    out.rip = rip;
    if (px.lock)         out.flags |= INSN_FLAG_LOCK;
    if (px.rep)          out.flags |= INSN_FLAG_REP;
    if (px.repne)        out.flags |= INSN_FLAG_REPNE;
    if (px.has_seg)      out.flags |= INSN_FLAG_HAS_SEG;
    if (px.opsize_pfx)   out.flags |= INSN_FLAG_OPSIZE_PFX;
    if (px.addrsize_pfx) out.flags |= INSN_FLAG_ADDRSIZE_PFX;
    out.cond = Cc::None;

    if (Status s = decoder::decode_after_prefixes(bs, px, rip, out); fail(s)) {
        return s;
    }
    out.len = bs.consumed;
    return Status::Ok;
}

} // namespace

Status decode_one(MemoryProvider& mem, GuestAddr rip, Insn& out) noexcept {
    MemCtx ctx{&mem, rip};
    ByteSource bs{&mem_fetch, &ctx, 0};
    return decode_with_source(bs, rip, out);
}

Status decode_bytes(const u8* bytes, usize len, GuestAddr rip, Insn& out) noexcept {
    if (bytes == nullptr) return Status::InvalidArgument;
    if (len == 0)         return Status::TruncatedInstruction;
    if (len > 15)         len = 15;  // hard cap per x86 spec
    BufCtx ctx{bytes, len};
    ByteSource bs{&buf_fetch, &ctx, 0};
    return decode_with_source(bs, rip, out);
}

// ---- Exposed to the per-opcode TU -----------------------------------------
namespace decoder {

Status read_byte_ext   (ByteSource& bs, u8&  out) noexcept { return read_byte(bs, out); }
Status read_u16_ext    (ByteSource& bs, u16& out) noexcept { return read_u16 (bs, out); }
Status read_u32_ext    (ByteSource& bs, u32& out) noexcept { return read_u32 (bs, out); }
Status read_u64_ext    (ByteSource& bs, u64& out) noexcept { return read_u64 (bs, out); }
Status peek_byte_ext   (ByteSource& bs, u8 off, u8& out) noexcept { return peek_byte(bs, off, out); }

} // namespace decoder

} // namespace emu
