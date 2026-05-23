// Private to the decoder TUs. Not exported.

#pragma once

#include "emu/error.h"
#include "emu/ir.h"
#include "emu/types.h"

namespace emu {

// A pull source of guest bytes. Used by the decoder to walk an instruction
// without caring whether bytes come from a vector (unit tests) or a
// MemoryProvider (live emulation). `consumed` is the running insn length.
struct ByteSource {
    Status (*fetch)(void* ctx, u8 off, u8& out);
    void*  ctx;
    u8     consumed;
};

struct Prefixes {
    bool opsize_pfx   = false;   // 0x66 (or VEX.pp == 01)
    bool addrsize_pfx = false;   // 0x67
    bool lock         = false;
    bool rep          = false;   // F3  (or VEX.pp == 10)
    bool repne        = false;   // F2  (or VEX.pp == 11)
    bool has_seg      = false;
    Seg  seg          = Seg::DS;
    bool has_rex      = false;
    u8   rex          = 0;

    // VEX prefix state. When `is_vex` is set, the next byte read is the
    // opcode in the implicit map selected by `vex_map` (1=0F, 2=0F38, 3=0F3A).
    // `vex_vvvv` is the *decoded* (already-inverted) 4-bit "second source"
    // register selector (0..15). `vex_l` is operand size: false=128 (xmm),
    // true=256 (ymm). The R/X/B extension bits are merged into the rex_*
    // accessors so per-op code can ignore the difference between REX and VEX.
    bool is_vex   = false;
    bool vex_w    = false;
    bool vex_l    = false;       // 0 -> 128-bit (xmm), 1 -> 256-bit (ymm)
    u8   vex_vvvv = 0;
    u8   vex_map  = 0;           // 1, 2, or 3

    [[nodiscard]] bool rex_w() const noexcept { return is_vex ? vex_w : (rex & 0x08) != 0; }
    [[nodiscard]] bool rex_r() const noexcept { return is_vex ? vex_r_ : (rex & 0x04) != 0; }
    [[nodiscard]] bool rex_x() const noexcept { return is_vex ? vex_x_ : (rex & 0x02) != 0; }
    [[nodiscard]] bool rex_b() const noexcept { return is_vex ? vex_b_ : (rex & 0x01) != 0; }

    // VEX REX-equivalent bits (internal; set by scan_prefixes).
    bool vex_r_ = false;
    bool vex_x_ = false;
    bool vex_b_ = false;

    // EVEX (AVX-512) prefix state. When `is_evex` is set, `is_vex` is also
    // set so the VEX-aware decoder paths still fire; vex_w/vex_l/vex_vvvv
    // are filled equivalently. EVEX-specific fields:
    //   evex_l2   -- high vector-length bit  (combined with vex_l: 00=128, 01=256, 10=512)
    //   evex_v_   -- V' bit  (high bit of 5-bit vvvv extension)
    //   evex_r_   -- R' bit  (high bit of dst-reg extension for ZMM16..31)
    //   evex_z    -- zeroing-vs-merging masking
    //   evex_b    -- broadcast/rounding-control/SAE
    //   evex_aaa  -- opmask register selector (0..7; 0 = no mask)
    bool is_evex   = false;
    bool evex_l2   = false;
    bool evex_v_   = false;
    bool evex_r_   = false;
    bool evex_z    = false;
    bool evex_b    = false;
    u8   evex_aaa  = 0;

    [[nodiscard]] u8  evex_vector_size() const noexcept {
        if (evex_l2) return 64;     // ZMM
        return vex_l ? 32 : 16;      // YMM or XMM
    }
};

// ModRM decoded fields.
struct ModRm {
    u8 mod;     // 0..3
    u8 reg;     // 0..7 (combine with REX.R for final reg id)
    u8 rm;      // 0..7 (combine with REX.B for final r/m reg id)
};

inline ModRm split_modrm(u8 b) noexcept {
    return ModRm{ static_cast<u8>(b >> 6), static_cast<u8>((b >> 3) & 7), static_cast<u8>(b & 7) };
}

// Effective operand size for general arithmetic / data-movement opcodes:
//   REX.W ⇒ 8 bytes
//   else 0x66 prefix ⇒ 2 bytes
//   else ⇒ 4 bytes (32-bit default in long mode)
inline u8 default_gpr_op_size(const Prefixes& px) noexcept {
    if (px.rex_w())     return 8;
    if (px.opsize_pfx)  return 2;
    return 4;
}

// Decode the r/m operand into `out`. If mod==3 this is a register operand;
// otherwise it's [base + index*scale + disp] (or RIP-relative). Reads any
// SIB byte and displacement following ModRM.
//
// `op_size` is the operand size (for memory access size; for register operands
// it's used to confirm we're not in 8-bit register weirdness territory).
Status decode_rm_operand(ByteSource& bs, const Prefixes& px, const ModRm& mrm,
                         u8 op_size, Operand& out) noexcept;

// Decode `reg` field into a `Reg` operand (combines with REX.R).
Operand reg_operand(const Prefixes& px, const ModRm& mrm, u8 op_size) noexcept;

// Read an immediate of `imm_size` bytes (1/2/4/8). Sign-extended to 64 bits
// for callers that want signed; raw for unsigned.
Status read_imm_sx(ByteSource& bs, u8 imm_size, i64& out) noexcept;

namespace decoder {
Status decode_after_prefixes(ByteSource& bs, Prefixes& px, GuestAddr rip, Insn& out) noexcept;

// Re-exported fetch helpers (so per-opcode TUs don't include decoder.cpp).
Status read_byte_ext (ByteSource& bs, u8&  out) noexcept;
Status read_u16_ext  (ByteSource& bs, u16& out) noexcept;
Status read_u32_ext  (ByteSource& bs, u32& out) noexcept;
Status read_u64_ext  (ByteSource& bs, u64& out) noexcept;
Status peek_byte_ext (ByteSource& bs, u8 off, u8& out) noexcept;
} // namespace decoder

} // namespace emu
