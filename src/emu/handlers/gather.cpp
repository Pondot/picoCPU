// AVX2 VGATHER family (VEX-encoded 0F 38 90..93).
//
// Layout (per `decode_vgather`):
//   dst.reg    = data destination XMM/YMM (0..15)
//   dst.index  = mask register (from VEX.vvvv) (0..15) -- high bit per lane gates load
//   src.kind   = Mem
//   src.reg    = base GPR (or NONE)
//   src.index  = vector index XMM/YMM register id (re-interpreted, NOT a GPR)
//   src.scale  = 1/2/4/8
//   src.imm    = sign-extended displacement
//   src.pad[0] = 1 (marker that src.index is a vector reg)
//   imm_extra low  byte = data lane size (4 or 8)
//   imm_extra next byte = index lane size (4 or 8)
//   op_size   = full vector width (16 = xmm, 32 = ymm)
//
// Per Intel semantics, after a successful gather, the mask lane is cleared
// (set to zero). Inactive lanes preserve the previous dst lane bits.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cstring>

namespace emu::handlers {

namespace {

// Read `n` bytes from a lane of `vec` at byte offset `off`, sign-extend to i64.
i64 read_index_lane(const u8* vec, u8 off, u8 n) noexcept {
    if (n == 4) {
        u32 v = 0;
        for (u8 i = 0; i < 4; ++i) v |= (u32(vec[off + i]) << (8 * i));
        return static_cast<i64>(static_cast<i32>(v));
    }
    // n == 8
    u64 v = 0;
    for (u8 i = 0; i < 8; ++i) v |= (u64(vec[off + i]) << (8 * i));
    return static_cast<i64>(v);
}

void do_gather(Cpu& cpu, const Insn& insn) noexcept {
    const u8 width       = insn.op_size;                 // 16 or 32
    const u8 data_lane   = static_cast<u8>(insn.imm_extra & 0xFF);
    const u8 idx_lane    = static_cast<u8>((insn.imm_extra >> 8) & 0xFF);
    const u8 num_lanes   = static_cast<u8>(width / data_lane);
    // Index register width = num_lanes * idx_lane bytes. May be a smaller
    // physical register than the data dest (e.g. VPGATHERDQ ymm has 4
    // qword data lanes but reads 4 dword indices from a 16-byte xmm view).
    const u8 idx_reg     = insn.src.index;
    const u8 mask_reg    = insn.dst.index;
    const u8 base_reg    = insn.src.reg;
    const u64 scale      = insn.src.scale ? insn.src.scale : 1;
    const i64 disp       = insn.src.imm;

    auto* mp = cpu.mem_read();
    if (!mp) {
        cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "vgather read");
        return;
    }

    u8 idx_vec[32]  = {};
    u8 mask_vec[32] = {};
    u8 dst_vec[32]  = {};
    std::memcpy(idx_vec,  cpu.ymm(idx_reg),  32);
    std::memcpy(mask_vec, cpu.ymm(mask_reg), 32);
    std::memcpy(dst_vec,  cpu.ymm(insn.dst.reg), 32);

    const u64 base_val = (base_reg == reg::NONE) ? 0ull : cpu.r(base_reg);
    u8 seg_off = 0;
    GuestAddr seg_base = 0;
    switch (insn.src.seg) {
        case Seg::FS: seg_base = cpu.fs_base(); break;
        case Seg::GS: seg_base = cpu.gs_base(); break;
        default: break;
    }
    (void)seg_off;

    for (u8 lane = 0; lane < num_lanes; ++lane) {
        const u8 mask_off = static_cast<u8>(lane * data_lane);
        // High bit of the mask lane gates the gather (Intel: sign-bit semantics).
        const u8 sign_byte = mask_vec[mask_off + (data_lane - 1)];
        const bool active  = (sign_byte & 0x80) != 0;
        if (!active) continue;

        const u8 idx_off = static_cast<u8>(lane * idx_lane);
        const i64 idx    = read_index_lane(idx_vec, idx_off, idx_lane);
        const GuestAddr addr = seg_base + base_val + static_cast<u64>(idx) * scale + static_cast<u64>(disp);

        u8 buf[8] = {};
        if (Status s = mp->read(addr, data_lane, buf); fail(s)) {
            cpu.set_fault(FaultKind::PageFault, addr, s, "vgather lane read");
            return;
        }
        std::memcpy(dst_vec + mask_off, buf, data_lane);
        // Clear mask lane on successful gather.
        std::memset(mask_vec + mask_off, 0, data_lane);
    }

    std::memcpy(cpu.ymm(insn.dst.reg), dst_vec, width);
    if (width == 16) cpu.zero_ymm_upper(insn.dst.reg);
    std::memcpy(cpu.ymm(mask_reg), mask_vec, width);
    if (width == 16) cpu.zero_ymm_upper(mask_reg);
}

} // namespace

void op_vpgatherdd (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }
void op_vpgatherdq (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }
void op_vpgatherqd (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }
void op_vpgatherqq (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }
void op_vgatherdps (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }
void op_vgatherdpd (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }
void op_vgatherqps (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }
void op_vgatherqpd (Cpu& cpu, const Insn& insn) { do_gather(cpu, insn); }

namespace {

// EVEX SCATTER: lanes in dst.reg are stored to per-lane memory addresses.
// Mask is the opmask register selected by `insn.pad & 0x7` (k0 means
// "all lanes active" per AVX-512 convention). Same layout as do_gather:
//   imm_extra low byte  = data lane size
//   imm_extra next byte = index lane size
void do_scatter(Cpu& cpu, const Insn& insn) noexcept {
    const u8 width     = insn.op_size;
    const u8 data_lane = static_cast<u8>(insn.imm_extra & 0xFF);
    const u8 idx_lane  = static_cast<u8>((insn.imm_extra >> 8) & 0xFF);
    const u8 num_lanes = static_cast<u8>(width / data_lane);
    const u8 aaa       = static_cast<u8>(insn.pad & 0x7u);
    const u64 mask_bits = (aaa == 0) ? ~u64{0} : cpu.k(aaa);
    const u8 idx_reg   = insn.src.index;
    const u8 base_reg  = insn.src.reg;
    const u64 scale    = insn.src.scale ? insn.src.scale : 1;
    const i64 disp     = insn.src.imm;

    auto* mp = cpu.mem_write();
    if (!mp) {
        cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "vscatter write");
        return;
    }

    u8 src_vec[64] = {};
    u8 idx_vec[64] = {};
    std::memcpy(src_vec, cpu.zmm(insn.dst.reg), width);
    std::memcpy(idx_vec, cpu.zmm(idx_reg),      width);

    const u64 base_val = (base_reg == reg::NONE) ? 0ull : cpu.r(base_reg);
    GuestAddr seg_base = 0;
    switch (insn.src.seg) {
        case Seg::FS: seg_base = cpu.fs_base(); break;
        case Seg::GS: seg_base = cpu.gs_base(); break;
        default: break;
    }

    for (u8 lane = 0; lane < num_lanes; ++lane) {
        if (!((mask_bits >> lane) & 1ull)) continue;
        const u8 idx_off = static_cast<u8>(lane * idx_lane);
        i64 idx = 0;
        if (idx_lane == 4) {
            u32 v = 0;
            for (u8 i = 0; i < 4; ++i) v |= (u32(idx_vec[idx_off + i]) << (8 * i));
            idx = static_cast<i64>(static_cast<i32>(v));
        } else {
            u64 v = 0;
            for (u8 i = 0; i < 8; ++i) v |= (u64(idx_vec[idx_off + i]) << (8 * i));
            idx = static_cast<i64>(v);
        }
        const GuestAddr addr = seg_base + base_val + static_cast<u64>(idx) * scale + static_cast<u64>(disp);
        if (Status s = mp->write(addr, data_lane, src_vec + lane * data_lane); fail(s)) {
            cpu.set_fault(FaultKind::PageFault, addr, s, "vscatter lane write");
            return;
        }
        // Per spec the mask lane is cleared on successful store; AVX-512
        // scatter uses opmask, so we clear the bit in the k register.
        if (aaa != 0) cpu.set_k(aaa, cpu.k(aaa) & ~(u64{1} << lane));
    }
}

} // namespace

void op_vpscatterdd(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }
void op_vpscatterdq(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }
void op_vpscatterqd(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }
void op_vpscatterqq(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }
void op_vscatterdps(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }
void op_vscatterdpd(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }
void op_vscatterqps(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }
void op_vscatterqpd(Cpu& cpu, const Insn& insn) { do_scatter(cpu, insn); }

} // namespace emu::handlers
