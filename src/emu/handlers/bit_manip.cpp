// Bit-manipulation handlers: BSF, BSR, BT/BTS/BTR/BTC, POPCNT, LZCNT, TZCNT.
//
// All operate on the operand width carried by `insn.op_size`. We use MSVC
// intrinsics where helpful (`_BitScanForward64`, `__popcnt64`, ...) and a
// portable fallback otherwise.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <intrin.h>

namespace emu::handlers {

namespace {

u8 popcount_u64(u64 v) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<u8>(__popcnt64(v));
#else
    u8 n = 0;
    while (v) { n += (v & 1); v >>= 1; }
    return n;
#endif
}

// First-set-bit index (LSB-first). Returns width when v == 0.
u8 tzcount_u64(u64 v, u8 width) noexcept {
    if (v == 0) return width;
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long idx = 0;
    _BitScanForward64(&idx, v);
    return static_cast<u8>(idx);
#else
    u8 n = 0;
    while (((v >> n) & 1ull) == 0 && n < width) ++n;
    return n;
#endif
}

// First-set-bit index (MSB-first). Returns width when v == 0.
u8 lzcount_u64(u64 v, u8 width) noexcept {
    if (v == 0) return width;
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long idx = 0;
    _BitScanReverse64(&idx, v);
    return static_cast<u8>(width - 1u - static_cast<u8>(idx));
#else
    u8 n = 0;
    while (((v >> (width - 1u - n)) & 1ull) == 0 && n < width) ++n;
    return n;
#endif
}

} // namespace

// BSF dst, src  ->  dst = index of lowest set bit in src; ZF=1 if src==0,
// dst undefined. Real CPUs leave dst unchanged when src==0; we mirror that.
void op_bsf(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, src)) return;
    cpu.force_sync_flags();
    if (src == 0) {
        cpu.set_zf(true);
        return;
    }
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u64 idx = tzcount_u64(src, w);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, idx);
    cpu.set_zf(false);
}

// BSR dst, src  ->  dst = index of highest set bit in src.
void op_bsr(Cpu& cpu, const Insn& insn) {
    u64 src = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, src)) return;
    cpu.force_sync_flags();
    if (src == 0) {
        cpu.set_zf(true);
        return;
    }
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u8 lead = lzcount_u64(src, w);
    const u64 idx = static_cast<u64>(w - 1 - lead);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, idx);
    cpu.set_zf(false);
}

// Bit-test family. Bit index = src (mod operand width when src is a reg;
// arbitrary for memory ops on real hw but we mask to width for safety).
namespace {

bool bit_test_get(Cpu& cpu, const Insn& insn, u64& base_val, u64& bit_off) noexcept {
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, base_val)) return false;
    u64 raw = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, raw)) return false;
    const u64 w = static_cast<u64>(insn.op_size * 8);
    bit_off = raw % w;
    return true;
}

} // namespace

void op_bt(Cpu& cpu, const Insn& insn) {
    u64 v = 0, bit = 0;
    if (!bit_test_get(cpu, insn, v, bit)) return;
    cpu.force_sync_flags();
    cpu.set_cf(((v >> bit) & 1ull) != 0);
}

void op_bts(Cpu& cpu, const Insn& insn) {
    u64 v = 0, bit = 0;
    if (!bit_test_get(cpu, insn, v, bit)) return;
    cpu.force_sync_flags();
    cpu.set_cf(((v >> bit) & 1ull) != 0);
    const u64 nv = v | (1ull << bit);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, nv);
}

void op_btr(Cpu& cpu, const Insn& insn) {
    u64 v = 0, bit = 0;
    if (!bit_test_get(cpu, insn, v, bit)) return;
    cpu.force_sync_flags();
    cpu.set_cf(((v >> bit) & 1ull) != 0);
    const u64 nv = v & ~(1ull << bit);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, nv);
}

void op_btc(Cpu& cpu, const Insn& insn) {
    u64 v = 0, bit = 0;
    if (!bit_test_get(cpu, insn, v, bit)) return;
    cpu.force_sync_flags();
    cpu.set_cf(((v >> bit) & 1ull) != 0);
    const u64 nv = v ^ (1ull << bit);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, nv);
}

void op_popcnt(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
    const u64 r = popcount_u64(v & mask_for(insn.op_size));
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    // POPCNT: ZF = (result == 0); CF=OF=SF=AF=PF=0.
    cpu.force_sync_flags();
    cpu.set_zf(r == 0);
    cpu.set_cf(false);
    cpu.set_of(false);
    cpu.set_sf(false);
    cpu.set_af(false);
    cpu.set_pf(false);
}

void op_lzcnt(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u8 c = lzcount_u64(v & mask_for(insn.op_size), w);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, c);
    cpu.force_sync_flags();
    cpu.set_cf(c == w);
    cpu.set_zf(c == 0);
}

void op_tzcnt(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
    const u8 w = static_cast<u8>(insn.op_size * 8);
    const u8 c = tzcount_u64(v & mask_for(insn.op_size), w);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, c);
    cpu.force_sync_flags();
    cpu.set_cf(c == w);
    cpu.set_zf(c == 0);
}

} // namespace emu::handlers
