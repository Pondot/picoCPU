// AVX-512 handlers. EVEX semantics:
//   dst  = insn.dst.reg                 (ZMM/YMM/XMM, 0..31)
//   src1 = insn.dst.index               (vvvv extended to 5 bits)
//   src2 = insn.src                     (ZMM reg or memory)
//   mask = insn.pad bits 0..2           (k0..k7; k0 means "no masking")
//   zero = insn.pad bit 3               (zeroing vs merging)
//   width = insn.op_size                (16, 32, or 64 -- XMM, YMM, ZMM)
//
// Per-lane semantics: for each lane index i,
//   active = mask[i] (or "always active" when k0)
//   if active:                      dst[i] = src1[i] OP src2[i]
//   else if zero:                   dst[i] = 0
//   else (merge):                   dst[i] unchanged
//
// EVEX-128 and EVEX-256 zero the upper ZMM half (bytes beyond width).

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cfenv>
#include <cstring>

namespace emu::handlers {

namespace {

bool read_vec(Cpu& cpu, const Insn& insn, const Operand& op, u8 width, u8* out) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(out, cpu.zmm(op.reg), width);
        return true;
    }
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "evex read"); return false; }
    const GuestAddr addr = ea(cpu, insn, op);
    if (Status s = mp->read(addr, width, out); fail(s)) {
        cpu.set_fault(FaultKind::PageFault, addr, s, "evex read");
        return false;
    }
    return true;
}

bool write_vec(Cpu& cpu, const Insn& insn, u8 reg_idx, u8 width, const u8* v) noexcept {
    // EVEX writes always go to a ZMM register; memory destinations are
    // unusual for the ops we cover and skipped.
    std::memcpy(cpu.zmm(reg_idx), v, width);
    cpu.zero_zmm_upper(reg_idx, width);
    (void)insn;
    return true;
}

// Mask helpers. `mask_reg = 0` ⇒ all lanes active. Otherwise bit i of k[mask_reg]
// gates lane i.
inline bool lane_active(u64 mask_bits, u8 lane_idx) noexcept {
    return (mask_bits >> lane_idx) & 1ull;
}

// Apply per-lane operation with masking.
template <typename Op>
void masked_binop(Cpu& cpu, const Insn& insn, u8 lane_size, Op op) noexcept {
    const u8 width   = insn.op_size;        // 16, 32, 64
    const u8 lanes   = static_cast<u8>(width / lane_size);
    const u8 aaa     = static_cast<u8>(insn.pad & 0x7u);
    const bool zero  = (insn.pad & 0x8u) != 0;
    const u64 mask_bits = (aaa == 0) ? ~u64{0} : cpu.k(aaa);

    u8 src1[64] = {}, src2[64] = {}, prev[64] = {};
    std::memcpy(src1, cpu.zmm(insn.dst.index), width);
    if (!read_vec(cpu, insn, insn.src, width, src2)) return;
    // Read previous dst for merging semantics.
    std::memcpy(prev, cpu.zmm(insn.dst.reg), width);

    u8 out[64] = {};
    for (u8 i = 0; i < lanes; ++i) {
        u8* o = out + (i * lane_size);
        const u8* a = src1 + (i * lane_size);
        const u8* b = src2 + (i * lane_size);
        const u8* p = prev + (i * lane_size);
        if (lane_active(mask_bits, i)) {
            op(a, b, o, lane_size);
        } else if (zero) {
            std::memset(o, 0, lane_size);
        } else {
            std::memcpy(o, p, lane_size);
        }
    }
    (void)write_vec(cpu, insn, insn.dst.reg, width, out);
}

// Masked move (2-operand): src -> dst with per-lane masking.
template <int LaneSize>
void masked_move(Cpu& cpu, const Insn& insn) noexcept {
    const u8 width   = insn.op_size;
    const u8 lanes   = static_cast<u8>(width / LaneSize);
    const u8 aaa     = static_cast<u8>(insn.pad & 0x7u);
    const bool zero  = (insn.pad & 0x8u) != 0;
    const u64 mask_bits = (aaa == 0) ? ~u64{0} : cpu.k(aaa);

    u8 src[64] = {}, prev[64] = {};
    if (!read_vec(cpu, insn, insn.src, width, src)) return;
    std::memcpy(prev, cpu.zmm(insn.dst.reg), width);

    u8 out[64] = {};
    for (u8 i = 0; i < lanes; ++i) {
        u8* o = out + (i * LaneSize);
        if (lane_active(mask_bits, i)) {
            std::memcpy(o, src + (i * LaneSize), LaneSize);
        } else if (zero) {
            std::memset(o, 0, LaneSize);
        } else {
            std::memcpy(o, prev + (i * LaneSize), LaneSize);
        }
    }
    (void)write_vec(cpu, insn, insn.dst.reg, width, out);
}

} // namespace

// ---- Bitwise ops (per-lane width matters only with masking) ----------------

void op_vpxord(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 4, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] ^ b[i]);
    });
}
void op_vpxorq(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 8, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] ^ b[i]);
    });
}
void op_vpandd(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 4, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] & b[i]);
    });
}
void op_vpandq(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 8, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] & b[i]);
    });
}
void op_vpord(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 4, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] | b[i]);
    });
}
void op_vporq(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 8, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] | b[i]);
    });
}

// ---- Packed integer add/sub ------------------------------------------------

void op_vpaddb_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 1, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] + b[i]);
    });
}
void op_vpaddw_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 2, [](const u8* a, const u8* b, u8* r, u8 n){
        const u16 av = u16(a[0]) | (u16(a[1]) << 8);
        const u16 bv = u16(b[0]) | (u16(b[1]) << 8);
        const u16 sum = static_cast<u16>(av + bv);
        r[0] = static_cast<u8>(sum & 0xFFu);
        r[1] = static_cast<u8>((sum >> 8) & 0xFFu);
        (void)n;
    });
}
void op_vpaddd_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 4, [](const u8* a, const u8* b, u8* r, u8 n){
        u32 av = 0, bv = 0;
        for (u8 i = 0; i < 4; ++i) { av |= (u32(a[i]) << (8*i)); bv |= (u32(b[i]) << (8*i)); }
        const u32 sum = av + bv;
        for (u8 i = 0; i < 4; ++i) r[i] = static_cast<u8>((sum >> (8*i)) & 0xFFu);
        (void)n;
    });
}
void op_vpaddq_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 8, [](const u8* a, const u8* b, u8* r, u8 n){
        u64 av = 0, bv = 0;
        for (u8 i = 0; i < 8; ++i) { av |= (u64(a[i]) << (8*i)); bv |= (u64(b[i]) << (8*i)); }
        const u64 sum = av + bv;
        for (u8 i = 0; i < 8; ++i) r[i] = static_cast<u8>((sum >> (8*i)) & 0xFFu);
        (void)n;
    });
}
void op_vpsubb_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 1, [](const u8* a, const u8* b, u8* r, u8 n){
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] - b[i]);
    });
}
void op_vpsubw_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 2, [](const u8* a, const u8* b, u8* r, u8 n){
        const u16 av = u16(a[0]) | (u16(a[1]) << 8);
        const u16 bv = u16(b[0]) | (u16(b[1]) << 8);
        const u16 d = static_cast<u16>(av - bv);
        r[0] = static_cast<u8>(d & 0xFFu);
        r[1] = static_cast<u8>((d >> 8) & 0xFFu);
        (void)n;
    });
}
void op_vpsubd_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 4, [](const u8* a, const u8* b, u8* r, u8 n){
        u32 av = 0, bv = 0;
        for (u8 i = 0; i < 4; ++i) { av |= (u32(a[i]) << (8*i)); bv |= (u32(b[i]) << (8*i)); }
        const u32 d = av - bv;
        for (u8 i = 0; i < 4; ++i) r[i] = static_cast<u8>((d >> (8*i)) & 0xFFu);
        (void)n;
    });
}
void op_vpsubq_evex(Cpu& cpu, const Insn& insn) {
    masked_binop(cpu, insn, 8, [](const u8* a, const u8* b, u8* r, u8 n){
        u64 av = 0, bv = 0;
        for (u8 i = 0; i < 8; ++i) { av |= (u64(a[i]) << (8*i)); bv |= (u64(b[i]) << (8*i)); }
        const u64 d = av - bv;
        for (u8 i = 0; i < 8; ++i) r[i] = static_cast<u8>((d >> (8*i)) & 0xFFu);
        (void)n;
    });
}

// ---- Masked moves ----------------------------------------------------------
void op_vmovdqu32(Cpu& cpu, const Insn& insn) { masked_move<4>(cpu, insn); }
void op_vmovdqu64(Cpu& cpu, const Insn& insn) { masked_move<8>(cpu, insn); }
void op_vmovdqa32(Cpu& cpu, const Insn& insn) { masked_move<4>(cpu, insn); }
void op_vmovdqa64(Cpu& cpu, const Insn& insn) { masked_move<8>(cpu, insn); }

// ---- FP per-lane apply with masking + broadcast ----------------------------
//
// `lane_size` is 4 (PS) or 8 (PD). When pad bit 4 (broadcast) is set and the
// source is a memory operand, only `lane_size` bytes are read and replicated
// across all lanes. Otherwise the full vector is read.

namespace {

bool read_vec_or_broadcast(Cpu& cpu, const Insn& insn, const Operand& op,
                           u8 width, u8 lane_size, u8* out) noexcept {
    const bool bcast = (insn.pad & 0x10u) != 0;
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(out, cpu.zmm(op.reg), width);
        return true;
    }
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "evex fp read"); return false; }
    const GuestAddr addr = ea(cpu, insn, op);
    if (bcast) {
        u8 lane_buf[8] = {};
        if (Status s = mp->read(addr, lane_size, lane_buf); fail(s)) {
            cpu.set_fault(FaultKind::PageFault, addr, s, "evex bcast read");
            return false;
        }
        const u8 lanes = static_cast<u8>(width / lane_size);
        for (u8 i = 0; i < lanes; ++i) std::memcpy(out + i * lane_size, lane_buf, lane_size);
        return true;
    }
    if (Status s = mp->read(addr, width, out); fail(s)) {
        cpu.set_fault(FaultKind::PageFault, addr, s, "evex fp read");
        return false;
    }
    return true;
}

// Scoped FP rounding-mode override. EVEX.b on a reg-reg FP op means SAE/RC:
// L'L bits become RC (00=RNE, 01=RD, 10=RU, 11=RZ). We snapshot the host's
// rounding mode, switch to the requested mode, and restore on dtor.
class ScopedFpRound {
public:
    explicit ScopedFpRound(const Insn& insn, bool reg_src) noexcept {
        if (reg_src && (insn.pad & 0x10u)) {
            const u8 rc = static_cast<u8>((insn.pad >> 5) & 0x3u);
            old_ = std::fegetround();
            switch (rc) {
                case 0: std::fesetround(FE_TONEAREST);   active_ = true; break;
                case 1: std::fesetround(FE_DOWNWARD);    active_ = true; break;
                case 2: std::fesetround(FE_UPWARD);      active_ = true; break;
                case 3: std::fesetround(FE_TOWARDZERO);  active_ = true; break;
            }
        }
    }
    ~ScopedFpRound() noexcept { if (active_) std::fesetround(old_); }
private:
    int  old_    = 0;
    bool active_ = false;
};

template <typename T, typename Op>
void fp_binop(Cpu& cpu, const Insn& insn, Op op) noexcept {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "FP lane must be 4 or 8 bytes");
    ScopedFpRound _rc(insn, insn.src.kind == OperandKind::Xmm);
    const u8 width      = insn.op_size;
    const u8 lane_size  = static_cast<u8>(sizeof(T));
    const u8 lanes      = static_cast<u8>(width / lane_size);
    const u8 aaa        = static_cast<u8>(insn.pad & 0x7u);
    const bool zero     = (insn.pad & 0x8u) != 0;
    const u64 mask_bits = (aaa == 0) ? ~u64{0} : cpu.k(aaa);

    u8 src1[64] = {}, src2[64] = {}, prev[64] = {};
    std::memcpy(src1, cpu.zmm(insn.dst.index), width);
    if (!read_vec_or_broadcast(cpu, insn, insn.src, width, lane_size, src2)) return;
    std::memcpy(prev, cpu.zmm(insn.dst.reg), width);

    u8 out[64] = {};
    for (u8 i = 0; i < lanes; ++i) {
        u8* o = out + (i * lane_size);
        if (lane_active(mask_bits, i)) {
            T a, b, r;
            std::memcpy(&a, src1 + i * lane_size, lane_size);
            std::memcpy(&b, src2 + i * lane_size, lane_size);
            r = op(a, b);
            std::memcpy(o, &r, lane_size);
        } else if (zero) {
            std::memset(o, 0, lane_size);
        } else {
            std::memcpy(o, prev + i * lane_size, lane_size);
        }
    }
    (void)write_vec(cpu, insn, insn.dst.reg, width, out);
}

template <typename T, typename Op>
void fp_unop(Cpu& cpu, const Insn& insn, Op op) noexcept {
    const u8 width      = insn.op_size;
    const u8 lane_size  = static_cast<u8>(sizeof(T));
    const u8 lanes      = static_cast<u8>(width / lane_size);
    const u8 aaa        = static_cast<u8>(insn.pad & 0x7u);
    const bool zero     = (insn.pad & 0x8u) != 0;
    const u64 mask_bits = (aaa == 0) ? ~u64{0} : cpu.k(aaa);

    u8 src[64] = {}, prev[64] = {};
    if (!read_vec_or_broadcast(cpu, insn, insn.src, width, lane_size, src)) return;
    std::memcpy(prev, cpu.zmm(insn.dst.reg), width);

    u8 out[64] = {};
    for (u8 i = 0; i < lanes; ++i) {
        u8* o = out + (i * lane_size);
        if (lane_active(mask_bits, i)) {
            T a, r;
            std::memcpy(&a, src + i * lane_size, lane_size);
            r = op(a);
            std::memcpy(o, &r, lane_size);
        } else if (zero) {
            std::memset(o, 0, lane_size);
        } else {
            std::memcpy(o, prev + i * lane_size, lane_size);
        }
    }
    (void)write_vec(cpu, insn, insn.dst.reg, width, out);
}

} // namespace

#include <cmath>

void op_vaddps_evex(Cpu& cpu, const Insn& insn) { fp_binop<float >(cpu, insn, [](float  a, float  b){ return a + b; }); }
void op_vaddpd_evex(Cpu& cpu, const Insn& insn) { fp_binop<double>(cpu, insn, [](double a, double b){ return a + b; }); }
void op_vsubps_evex(Cpu& cpu, const Insn& insn) { fp_binop<float >(cpu, insn, [](float  a, float  b){ return a - b; }); }
void op_vsubpd_evex(Cpu& cpu, const Insn& insn) { fp_binop<double>(cpu, insn, [](double a, double b){ return a - b; }); }
void op_vmulps_evex(Cpu& cpu, const Insn& insn) { fp_binop<float >(cpu, insn, [](float  a, float  b){ return a * b; }); }
void op_vmulpd_evex(Cpu& cpu, const Insn& insn) { fp_binop<double>(cpu, insn, [](double a, double b){ return a * b; }); }
void op_vdivps_evex(Cpu& cpu, const Insn& insn) { fp_binop<float >(cpu, insn, [](float  a, float  b){ return a / b; }); }
void op_vdivpd_evex(Cpu& cpu, const Insn& insn) { fp_binop<double>(cpu, insn, [](double a, double b){ return a / b; }); }

void op_vminps_evex(Cpu& cpu, const Insn& insn) {
    fp_binop<float>(cpu, insn, [](float a, float b){
        // Intel: if either is NaN or both are zero, return second source.
        if (std::isnan(a) || std::isnan(b)) return b;
        if (a == 0.0f && b == 0.0f) return b;
        return a < b ? a : b;
    });
}
void op_vminpd_evex(Cpu& cpu, const Insn& insn) {
    fp_binop<double>(cpu, insn, [](double a, double b){
        if (std::isnan(a) || std::isnan(b)) return b;
        if (a == 0.0 && b == 0.0) return b;
        return a < b ? a : b;
    });
}
void op_vmaxps_evex(Cpu& cpu, const Insn& insn) {
    fp_binop<float>(cpu, insn, [](float a, float b){
        if (std::isnan(a) || std::isnan(b)) return b;
        if (a == 0.0f && b == 0.0f) return b;
        return a > b ? a : b;
    });
}
void op_vmaxpd_evex(Cpu& cpu, const Insn& insn) {
    fp_binop<double>(cpu, insn, [](double a, double b){
        if (std::isnan(a) || std::isnan(b)) return b;
        if (a == 0.0 && b == 0.0) return b;
        return a > b ? a : b;
    });
}

void op_vsqrtps_evex(Cpu& cpu, const Insn& insn) { fp_unop<float >(cpu, insn, [](float  a){ return std::sqrt(a); }); }
void op_vsqrtpd_evex(Cpu& cpu, const Insn& insn) { fp_unop<double>(cpu, insn, [](double a){ return std::sqrt(a); }); }

} // namespace emu::handlers
