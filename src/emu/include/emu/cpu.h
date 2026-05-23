// Emulator CPU state.
//
// 16 GPRs plus RIP and RFLAGS. Eager-flags first (correctness > performance);
// the lazy `cc_op`/`cc_src`/`cc_dst`/`cc_src2` model is reserved fields here
// and will replace the eager path in Phase 2 once the diff harness is solid.

#pragma once

#include "emu/error.h"
#include "emu/fault.h"
#include "emu/ir.h"
#include "emu/types.h"

#include <vector>

namespace emu {

class MemoryProvider;
class HookManager;

// ---- Lazy-EFLAGS model ------------------------------------------------------
//
// Borrowed in shape from QEMU's `qemu/target/i386/cpu.h` (CC_OP_*). After
// each flag-producing op we stash:
//   cc_op   : which op produced the flags (Add / Sub / Logic / ...)
//   cc_size : 1, 2, 4, or 8 (operand width in bytes)
//   cc_src  : the "right-hand" operand (b in `a + b`, count in shifts, etc.)
//   cc_dst  : the result
//   cc_src2 : an extra source (e.g. carry-in for ADC/SBB)
//
// When code reads a flag (cf(), zf(), etc.) we lazily compute *all six*
// flags from cc_*, write them into rflags_, then transition cc_op to Eflags
// so subsequent reads short-circuit.
//
// Recovering the original first operand for SUB/ADD: a = cc_dst ± cc_src.
enum class CcOp : u16 {
    Eflags = 0,   // rflags_ is authoritative -- no recompute needed
    Add,          // cc_dst = a + b,         cc_src = b
    Adc,          // cc_dst = a + b + cin,   cc_src = b, cc_src2 = cin
    Sub,          // cc_dst = a - b,         cc_src = b
    Sbb,          // cc_dst = a - b - bin,   cc_src = b, cc_src2 = bin
    Inc,          // cc_dst = a + 1,         cc_src = saved CF
    Dec,          // cc_dst = a - 1,         cc_src = saved CF
    Neg,          // cc_dst = -a
    Logic,        // cc_dst = result;  CF=OF=0, AF undefined->cleared
    Shl,          // cc_dst = result, cc_src = original, cc_src2 = (count & mask)
    Shr,          // logical shift right; same fields as Shl
    Sar,          // arithmetic shift right
    Rol,          // cc_dst = result, cc_src = original, cc_src2 = effective count
    Ror,          //
    MulOv,        // 2/3-operand IMUL: CF=OF=cc_src (overflow bit). ZSP undefined.
};

namespace rflags {
    constexpr u64 CF   = 1ull << 0;   // carry
    constexpr u64 PF   = 1ull << 2;   // parity
    constexpr u64 AF   = 1ull << 4;   // aux carry (BCD)
    constexpr u64 ZF   = 1ull << 6;   // zero
    constexpr u64 SF   = 1ull << 7;   // sign
    constexpr u64 TF   = 1ull << 8;
    constexpr u64 IF   = 1ull << 9;
    constexpr u64 DF   = 1ull << 10;  // direction (1 = down)
    constexpr u64 OF   = 1ull << 11;  // overflow
    constexpr u64 IOPL = 3ull << 12;
    constexpr u64 NT   = 1ull << 14;
    constexpr u64 RF   = 1ull << 16;
    constexpr u64 VM   = 1ull << 17;
    constexpr u64 AC   = 1ull << 18;
    constexpr u64 VIF  = 1ull << 19;
    constexpr u64 VIP  = 1ull << 20;
    constexpr u64 ID   = 1ull << 21;

    // Bits that arithmetic ops are responsible for.
    constexpr u64 ARITH_MASK = CF | PF | AF | ZF | SF | OF;
    // Reserved bit 1 is always 1 in real RFLAGS. We honor that to keep parity
    // with what `pushfq` would observe natively.
    constexpr u64 ALWAYS_ONE = 1ull << 1;
}

class Cpu {
public:
    // ---- GPRs -----------------------------------------------------------
    [[nodiscard]] u64 r(u8 idx) const noexcept { return gpr_[idx]; }
    void              set_r(u8 idx, u64 v) noexcept { gpr_[idx] = v; }

    // Sized accessors. Writing a 32-bit register zero-extends to the full 64
    // (x86-64 rule). 8/16 leave the upper bits alone.
    //
    // High-byte aliases (AH, CH, DH, BH -- accessed when ModRM.reg/rm ∈ {4..7}
    // for an 8-bit operand AND no REX prefix is present) are encoded as
    // indices 16..19. The decoder is responsible for translating reg=4..7
    // -> idx=16..19 when REX is absent; the handler call sites just see the
    // 16..19 index and these accessors do the right shift.
    [[nodiscard]] u8  r8 (u8 idx) const noexcept {
        if (idx >= 16 && idx < 20) {
            return static_cast<u8>((gpr_[idx - 16] >> 8) & 0xFFull);
        }
        return static_cast<u8>(gpr_[idx] & 0xFFull);
    }
    [[nodiscard]] u16 r16(u8 idx) const noexcept { return static_cast<u16>(gpr_[idx] & 0xFFFFull); }
    [[nodiscard]] u32 r32(u8 idx) const noexcept { return static_cast<u32>(gpr_[idx] & 0xFFFFFFFFull); }

    void set_r8 (u8 idx, u8  v) noexcept {
        if (idx >= 16 && idx < 20) {
            const u8 base = idx - 16;
            gpr_[base] = (gpr_[base] & ~u64{0xFF00ull}) | (u64{v} << 8);
            return;
        }
        gpr_[idx] = (gpr_[idx] & ~0xFFull) | u64{v};
    }
    void set_r16(u8 idx, u16 v) noexcept { gpr_[idx] = (gpr_[idx] & ~0xFFFFull)   | u64{v}; }
    void set_r32(u8 idx, u32 v) noexcept { gpr_[idx] = u64{v}; }
    void set_r64(u8 idx, u64 v) noexcept { gpr_[idx] = v; }

    // ---- RIP ------------------------------------------------------------
    [[nodiscard]] GuestAddr rip() const noexcept { return rip_; }
    void                    set_rip(GuestAddr v) noexcept { rip_ = v; }

    // ---- RFLAGS ---------------------------------------------------------
    //
    // All readers go through `sync_flags_()` which lazily computes the six
    // arith flags from cc_op state on first access after a flag-producing op.
    [[nodiscard]] u64 rflags() const noexcept { sync_flags_(); return rflags_ | rflags::ALWAYS_ONE; }
    void              set_rflags(u64 v) noexcept { rflags_ = v; cc_op_ = CcOp::Eflags; }

    [[nodiscard]] bool cf() const noexcept { sync_flags_(); return (rflags_ & rflags::CF) != 0; }
    [[nodiscard]] bool pf() const noexcept { sync_flags_(); return (rflags_ & rflags::PF) != 0; }
    [[nodiscard]] bool af() const noexcept { sync_flags_(); return (rflags_ & rflags::AF) != 0; }
    [[nodiscard]] bool zf() const noexcept { sync_flags_(); return (rflags_ & rflags::ZF) != 0; }
    [[nodiscard]] bool sf() const noexcept { sync_flags_(); return (rflags_ & rflags::SF) != 0; }
    [[nodiscard]] bool of() const noexcept { sync_flags_(); return (rflags_ & rflags::OF) != 0; }
    [[nodiscard]] bool df() const noexcept { return (rflags_ & rflags::DF) != 0; }  // DF is not lazy

    // Direct flag writes -- used by ops like SAHF / POPF / explicit flag-setting
    // and by the eager-path cross-check tests. Force sync first, then write.
    void set_cf(bool v) noexcept { sync_flags_(); set_bit_(rflags::CF, v); }
    void set_pf(bool v) noexcept { sync_flags_(); set_bit_(rflags::PF, v); }
    void set_af(bool v) noexcept { sync_flags_(); set_bit_(rflags::AF, v); }
    void set_zf(bool v) noexcept { sync_flags_(); set_bit_(rflags::ZF, v); }
    void set_sf(bool v) noexcept { sync_flags_(); set_bit_(rflags::SF, v); }
    void set_of(bool v) noexcept { sync_flags_(); set_bit_(rflags::OF, v); }
    void set_df(bool v) noexcept { set_bit_(rflags::DF, v); }

    // ---- Lazy stash entry points ---------------------------------------
    //
    // Handlers call these after producing a result. They do *not* touch
    // rflags_. The flag accessors lazily compute when a flag is read.
    void stash_logic(u8 size, u64 r) noexcept;
    void stash_add  (u8 size, u64 b, u64 r) noexcept;
    void stash_adc  (u8 size, u64 b, bool cin, u64 r) noexcept;
    void stash_sub  (u8 size, u64 b, u64 r) noexcept;
    void stash_sbb  (u8 size, u64 b, bool bin, u64 r) noexcept;
    void stash_inc  (u8 size, u64 r) noexcept;
    void stash_dec  (u8 size, u64 r) noexcept;
    void stash_neg  (u8 size, u64 a, u64 r) noexcept;
    void stash_shl  (u8 size, u64 original, u8 count, u64 r) noexcept;
    void stash_shr  (u8 size, u64 original, u8 count, u64 r) noexcept;
    void stash_sar  (u8 size, u64 original, u8 count, u64 r) noexcept;
    void stash_rol  (u8 size, u64 original, u8 count, u64 r) noexcept;
    void stash_ror  (u8 size, u64 original, u8 count, u64 r) noexcept;
    void stash_mul_ov(bool overflow) noexcept;

    // Force `rflags_` to hold authoritative bits; subsequent reads short-circuit.
    void force_sync_flags() const noexcept { sync_flags_(); }

    // ---- Segment bases (Phase 2 will fill these from the host thread) ---
    [[nodiscard]] u64 fs_base() const noexcept { return fs_base_; }
    [[nodiscard]] u64 gs_base() const noexcept { return gs_base_; }
    void set_fs_base(u64 v) noexcept { fs_base_ = v; }
    void set_gs_base(u64 v) noexcept { gs_base_ = v; }

    // ---- x87 FPU stack (ST0..ST7) -------------------------------------
    //
    // Approximation: registers stored as 64-bit doubles, not the architectural
    // 80-bit extended. Adequate for the vast majority of compiler-emitted x87
    // (long-double-to-double conversions happen at function boundaries).
    //
    // `fpu_top_` is the TOP-of-stack pointer (0..7). ST(i) accesses the
    // physical register at index `(fpu_top_ + i) & 7`.
    [[nodiscard]] u8     fpu_top()        const noexcept { return fpu_top_ & 7; }
    [[nodiscard]] u16    fpu_cw()         const noexcept { return fpu_cw_; }
    [[nodiscard]] u16    fpu_sw()         const noexcept {
        return static_cast<u16>((fpu_sw_ & ~u16{0x3800}) | (static_cast<u16>(fpu_top_ & 7) << 11));
    }
    void                 set_fpu_cw(u16 v) noexcept { fpu_cw_ = v; }
    void                 set_fpu_sw(u16 v) noexcept {
        fpu_sw_  = static_cast<u16>(v & ~u16{0x3800});
        fpu_top_ = static_cast<u8>((v >> 11) & 7);
    }

    [[nodiscard]] double st(u8 i) const noexcept { return fpu_st_[(fpu_top_ + i) & 7]; }
    void                 set_st(u8 i, double v) noexcept { fpu_st_[(fpu_top_ + i) & 7] = v; }

    // Stack management. push: TOP--, ST(0) = v. pop: discard ST(0), TOP++.
    void fpu_push(double v) noexcept {
        fpu_top_ = static_cast<u8>((fpu_top_ - 1) & 7);
        fpu_st_[fpu_top_] = v;
    }
    double fpu_pop() noexcept {
        const double v = fpu_st_[fpu_top_];
        fpu_top_ = static_cast<u8>((fpu_top_ + 1) & 7);
        return v;
    }

    // ---- Debug registers (DR0-DR7) ------------------------------------
    //
    // DR0-DR3 hold breakpoint linear addresses; DR6 holds breakpoint hit
    // status; DR7 holds enable bits + breakpoint conditions. We expose them
    // via get/set; the dispatcher consults the enable bits to decide whether
    // to fire a #BP on each instruction.
    [[nodiscard]] u64 dr(u8 idx) const noexcept { return dr_[idx & 7]; }
    void              set_dr(u8 idx, u64 v) noexcept { dr_[idx & 7] = v; }

    // Returns true if any enabled DR0-DR3 breakpoint matches `rip` (execute-
    // type breakpoint; DR7 condition bits 16/20/24/28 == 0b00 for execute).
    // On match, sets the corresponding DR6.B<n> status bit (bits 0-3) and
    // the BS (single-step) bit clear, per Intel SDM #DB delivery semantics.
    [[nodiscard]] bool exec_breakpoint_at(GuestAddr rip) const noexcept {
        const u64 dr7 = dr_[7];
        for (int n = 0; n < 4; ++n) {
            const bool enabled = ((dr7 >> (2 * n)) & 0x3ull) != 0;
            const u8 cond = static_cast<u8>((dr7 >> (16 + 4 * n)) & 0x3ull);
            if (enabled && cond == 0b00 && dr_[n] == rip) {
                const_cast<Cpu*>(this)->dr_[6] |= (1ull << n);
                return true;
            }
        }
        return false;
    }

    // Data-access breakpoint match (DR0-DR3 with condition 01=write, 11=RW).
    // `is_write` distinguishes load (false) from store (true). `size` is the
    // access width in bytes. The length encoded in DR7 (bits 18+4n .. 19+4n)
    // is interpreted per Intel SDM: 00=1, 01=2, 10=8, 11=4.
    [[nodiscard]] bool data_breakpoint_at(GuestAddr addr, bool is_write, u8 size) const noexcept {
        const u64 dr7 = dr_[7];
        for (int n = 0; n < 4; ++n) {
            const bool enabled = ((dr7 >> (2 * n)) & 0x3ull) != 0;
            if (!enabled) continue;
            const u8 cond = static_cast<u8>((dr7 >> (16 + 4 * n)) & 0x3ull);
            // 00=exec (skip here), 01=store, 10=I/O (skip), 11=RW
            const bool matches_kind = (cond == 0b01 && is_write)
                                    || (cond == 0b11);
            if (!matches_kind) continue;
            const u8 len_field = static_cast<u8>((dr7 >> (18 + 4 * n)) & 0x3ull);
            static constexpr u8 LEN_TABLE[4] = {1, 2, 8, 4};
            const u8 bp_size = LEN_TABLE[len_field];
            const GuestAddr bp_lo = dr_[n];
            const GuestAddr bp_hi = bp_lo + bp_size;
            const GuestAddr ac_lo = addr;
            const GuestAddr ac_hi = addr + size;
            if (ac_lo < bp_hi && ac_hi > bp_lo) {
                // Set DR6.Bn to signal which breakpoint fired.
                const_cast<Cpu*>(this)->dr_[6] |= (1ull << n);
                return true;
            }
        }
        return false;
    }

    // ---- XMM / YMM / ZMM registers (SSE/AVX/AVX-512) -------------------
    //
    // 32 512-bit ZMM registers (AVX-512 extends from 16 -> 32). XMM = low 128
    // bits, YMM = low 256 bits. Storage is `zmm_[32][64]`.
    //
    // SSE ops leave bits beyond their width untouched. VEX-128 zeros the
    // upper YMM. EVEX-128/256 zeros the upper ZMM. The dispatcher/handler
    // helpers honor these rules.
    [[nodiscard]] const u8* xmm(u8 idx) const noexcept { return zmm_[idx]; }
    [[nodiscard]]       u8* xmm(u8 idx)       noexcept { return zmm_[idx]; }
    [[nodiscard]] const u8* ymm(u8 idx) const noexcept { return zmm_[idx]; }
    [[nodiscard]]       u8* ymm(u8 idx)       noexcept { return zmm_[idx]; }
    [[nodiscard]] const u8* zmm(u8 idx) const noexcept { return zmm_[idx]; }
    [[nodiscard]]       u8* zmm(u8 idx)       noexcept { return zmm_[idx]; }

    // VEX-128 / EVEX-128 zeroing: write the low N bytes (handler-driven),
    // then clear bytes [N, 64). Called from VEX/EVEX handlers after they
    // finish writing.
    void zero_ymm_upper(u8 idx) noexcept {
        for (int i = 16; i < 64; ++i) zmm_[idx][i] = 0;
    }
    void zero_zmm_upper(u8 idx, u8 width_bytes) noexcept {
        for (u8 i = width_bytes; i < 64; ++i) zmm_[idx][i] = 0;
    }

    [[nodiscard]] u64 xmm_qword(u8 idx, u8 lane) const noexcept {
        u64 v = 0;
        for (int i = 0; i < 8; ++i) v |= (u64{zmm_[idx][lane * 8 + i]} << (8 * i));
        return v;
    }
    void set_xmm_qword(u8 idx, u8 lane, u64 v) noexcept {
        for (int i = 0; i < 8; ++i) zmm_[idx][lane * 8 + i] = static_cast<u8>((v >> (8 * i)) & 0xFFu);
    }

    // AVX-512 opmask registers k0..k7. k0 is special -- when used as a mask
    // it means "no masking" (every lane active).
    [[nodiscard]] u64 k(u8 idx) const noexcept { return k_[idx & 7]; }
    void              set_k(u8 idx, u64 v) noexcept { k_[idx & 7] = v; }

    // ---- CPUID overrides ----------------------------------------------
    //
    // op_cpuid consults this table first; any (leaf, subleaf) match returns
    // the stored EAX/EBX/ECX/EDX directly. Misses fall back to the canned
    // "realistic Skylake-ish" defaults in op_cpuid.
    struct CpuidLeaf { u32 leaf, subleaf, eax, ebx, ecx, edx; };
    void set_cpuid(u32 leaf, u32 subleaf, u32 eax, u32 ebx, u32 ecx, u32 edx) noexcept {
        for (auto& e : cpuid_overrides_) {
            if (e.leaf == leaf && e.subleaf == subleaf) {
                e.eax = eax; e.ebx = ebx; e.ecx = ecx; e.edx = edx;
                return;
            }
        }
        cpuid_overrides_.push_back(CpuidLeaf{leaf, subleaf, eax, ebx, ecx, edx});
    }
    [[nodiscard]] bool cpuid_lookup(u32 leaf, u32 subleaf,
                                    u32& eax, u32& ebx, u32& ecx, u32& edx) const noexcept {
        for (const auto& e : cpuid_overrides_) {
            if (e.leaf == leaf && e.subleaf == subleaf) {
                eax = e.eax; ebx = e.ebx; ecx = e.ecx; edx = e.edx;
                return true;
            }
        }
        return false;
    }
    void clear_cpuid_overrides() noexcept { cpuid_overrides_.clear(); }

    // ---- Memory binding -----------------------------------------------
    //
    // The emulator sets these once before starting a run. Handlers read
    // through these for any memory operand. The provider may be a cache
    // chain (L1->L2->L3->Win backing) -- handlers don't care.
    void attach_memory_read (MemoryProvider* mp) noexcept { mem_read_  = mp; }
    void attach_memory_write(MemoryProvider* mp) noexcept { mem_write_ = mp; }
    [[nodiscard]] MemoryProvider* mem_read () const noexcept { return mem_read_; }
    [[nodiscard]] MemoryProvider* mem_write() const noexcept { return mem_write_; }

    void attach_hooks(HookManager* hm) noexcept { hooks_ = hm; }
    [[nodiscard]] HookManager* hooks() const noexcept { return hooks_; }

    // Latch a fault. The kind-aware overload is the canonical one; the
    // Status-only overload is a migration shim that maps via kind_from_status.
    void set_fault(FaultKind kind, GuestAddr at, Status reason = Status::Ok,
                   const char* note = nullptr) noexcept {
        fault_.kind   = kind;
        fault_.rip    = rip_;
        fault_.addr   = at;
        fault_.reason = reason;
        fault_.note   = note;
        halted_       = true;
    }
    void set_fault(GuestAddr at, Status reason) noexcept {
        set_fault(kind_from_status(reason), at, reason);
    }

    [[nodiscard]] const Fault& fault() const noexcept { return fault_; }
    void clear_fault() noexcept { fault_ = Fault{}; }
    [[nodiscard]] Status     fault_reason() const noexcept { return fault_.reason; }
    [[nodiscard]] GuestAddr  fault_at()     const noexcept { return fault_.addr; }

    // Fault hook: invoked once on the first latched fault.
    using FaultCb = void (*)(void* user, const Cpu& cpu, const Fault& fault);
    void set_fault_hook(FaultCb cb, void* user) noexcept { fault_cb_ = cb; fault_user_ = user; }
    [[nodiscard]] FaultCb fault_hook()      const noexcept { return fault_cb_; }
    [[nodiscard]] void*   fault_hook_user() const noexcept { return fault_user_; }

    // ---- Run state ------------------------------------------------------
    [[nodiscard]] bool halted() const noexcept { return halted_; }
    void               halt()         noexcept { halted_ = true; }
    void               unhalt()       noexcept { halted_ = false; }

    [[nodiscard]] bool branch_taken() const noexcept { return branch_taken_; }
    void               clear_branch()       noexcept { branch_taken_ = false; }
    // Called by control-flow handlers: signals the dispatcher to use this RIP
    // instead of advancing past the current instruction.
    void               take_branch(GuestAddr target) noexcept {
        rip_ = target;
        branch_taken_ = true;
    }

    // ---- Cond eval ------------------------------------------------------
    [[nodiscard]] bool cond_holds(Cc cc) const noexcept;

    // ---- Eager-flag helpers (cross-check only) -------------------------
    //
    // These are NOT called by handlers anymore -- handlers stash via the
    // `stash_*` API and let the reader compute lazily. Kept as a reference
    // implementation that the lazy path is validated against in unit tests.
    //
    // `size_bytes` is 1/2/4/8. Inputs are masked to that width internally.
    void set_flags_logical_eager(u8 size_bytes, u64 result) noexcept;
    void set_flags_add_eager    (u8 size_bytes, u64 a, u64 b, u64 result) noexcept;
    void set_flags_adc_eager    (u8 size_bytes, u64 a, u64 b, bool cin, u64 result) noexcept;
    void set_flags_sub_eager    (u8 size_bytes, u64 a, u64 b, u64 result) noexcept;
    void set_flags_sbb_eager    (u8 size_bytes, u64 a, u64 b, bool bin, u64 result) noexcept;
    void set_flags_inc_eager    (u8 size_bytes, u64 prev, u64 result) noexcept;
    void set_flags_dec_eager    (u8 size_bytes, u64 prev, u64 result) noexcept;
    void set_flags_neg_eager    (u8 size_bytes, u64 src, u64 result) noexcept;
    void set_flags_shl_eager    (u8 size_bytes, u64 src, u8 count, u64 result) noexcept;
    void set_flags_shr_eager    (u8 size_bytes, u64 src, u8 count, u64 result) noexcept;
    void set_flags_sar_eager    (u8 size_bytes, u64 src, u8 count, u64 result) noexcept;
    void set_flags_rol_eager    (u8 size_bytes, u64 src, u8 count, u64 result) noexcept;
    void set_flags_ror_eager    (u8 size_bytes, u64 src, u8 count, u64 result) noexcept;

    // Reset everything to zero (RFLAGS retains its always-on bit).
    void reset() noexcept;

    // Lightweight snapshot of CPU state (GPRs, RIP, RFLAGS, FS/GS, XMM/YMM,
    // x87 stack + CW/SW/TOP, DRs, lazy-flag stash, halt/branch flags).
    // Returns an opaque struct caller passes back to restore().
    struct Snapshot {
        u64       gpr[16];
        u64       dr[8];
        alignas(64) u8 zmm[32][64];
        u64       k[8];
        double    fpu_st[8];
        u16       fpu_cw;
        u16       fpu_sw;
        u8        fpu_top;
        GuestAddr rip;
        u64       rflags;
        u64       fs_base;
        u64       gs_base;
        CcOp      cc_op;
        u8        cc_size;
        u64       cc_dst, cc_src, cc_src2;
        Fault     fault;
        bool      halted;
        bool      branch_taken;
    };

    void snapshot(Snapshot& out) const noexcept;
    void restore(const Snapshot& s) noexcept;

private:
    void set_bit_(u64 mask, bool v) noexcept {
        if (v) rflags_ |= mask;
        else   rflags_ &= ~mask;
    }

    // Lazily compute arith flags from cc_op state. No-op when already synced.
    void sync_flags_() const noexcept;

    u64             gpr_[16] = {};      // 0..15
    u64             dr_[8] = {};         // DR0..DR7
    alignas(64) u8  zmm_[32][64] = {};  // ZMM0..ZMM31 (YMM = low 32, XMM = low 16)
    u64             k_[8] = {};           // k0..k7 opmask registers
    double          fpu_st_[8] = {};    // x87 stack registers (physical, not ST(i))
    u16             fpu_cw_  = 0x037F;  // control word default (per Intel SDM)
    u16             fpu_sw_  = 0;       // status word (low 8 bits + condition codes)
    u8              fpu_top_ = 0;       // TOP pointer 0..7
    GuestAddr       rip_         = 0;
    mutable u64     rflags_      = rflags::ALWAYS_ONE;  // mutable: sync writes through
    u64             fs_base_     = 0;
    u64             gs_base_     = 0;
    MemoryProvider* mem_read_    = nullptr;
    MemoryProvider* mem_write_   = nullptr;
    HookManager*    hooks_       = nullptr;
    Fault           fault_{};
    FaultCb         fault_cb_    = nullptr;
    void*           fault_user_  = nullptr;

    // Lazy-flag stash (see CcOp comment above for semantics).
    mutable CcOp    cc_op_    = CcOp::Eflags;
    u8              cc_size_  = 4;
    u64             cc_dst_   = 0;
    u64             cc_src_   = 0;
    u64             cc_src2_  = 0;

    bool            halted_       = false;
    bool            branch_taken_ = false;

    std::vector<CpuidLeaf> cpuid_overrides_;
};

// ---- Mask helpers -----------------------------------------------------------
// `mask_for(n)` returns the all-ones value for an n-byte width: 0xFF, 0xFFFF, ...
[[nodiscard]] inline u64 mask_for(u8 size_bytes) noexcept {
    switch (size_bytes) {
        case 1: return 0xFFull;
        case 2: return 0xFFFFull;
        case 4: return 0xFFFFFFFFull;
        case 8: return 0xFFFFFFFFFFFFFFFFull;
        default: return 0;
    }
}

[[nodiscard]] inline u8 sign_bit_pos(u8 size_bytes) noexcept {
    return static_cast<u8>(size_bytes * 8u - 1u);
}

[[nodiscard]] inline u64 sign_mask(u8 size_bytes) noexcept {
    return 1ull << sign_bit_pos(size_bytes);
}

} // namespace emu
