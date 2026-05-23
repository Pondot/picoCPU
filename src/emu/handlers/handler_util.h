// Shared internals for handler TUs.
//
// All operand resolution lives here so each handler reads like math, not
// like a switch over operand kinds. These functions are NOT exported.

#pragma once

#include "emu/cpu.h"
#include "emu/hooks.h"
#include "emu/ir.h"
#include "emu/memory.h"
#include "emu/types.h"

namespace emu::handlers {

// Compute effective address of a memory operand.
//
// For OperandKind::Mem:  base + index*scale + disp (segment override applies
// via cpu.fs_base()/gs_base() for FS/GS; DS/ES/SS/CS are flat at base 0 in
// long mode).
//
// For OperandKind::RipRelMem: (insn.rip + insn.len) + disp.
inline GuestAddr ea(const Cpu& cpu, const Insn& insn, const Operand& op) noexcept {
    if (op.kind == OperandKind::RipRelMem) {
        // RIP-relative: effective address is the RIP of the *next* instruction
        // plus the sign-extended disp. The dispatcher has not advanced RIP
        // yet when the handler runs, so we synthesize it from insn fields.
        return insn.rip + insn.len + static_cast<u64>(op.imm);
    }
    GuestAddr addr = 0;
    if (op.reg != reg::NONE)   addr += cpu.r(op.reg);
    if (op.index != reg::NONE) addr += cpu.r(op.index) * static_cast<u64>(op.scale);
    addr += static_cast<u64>(op.imm);

    switch (op.seg) {
        case Seg::FS: addr += cpu.fs_base(); break;
        case Seg::GS: addr += cpu.gs_base(); break;
        default: break;
    }
    return addr;
}

// Read an operand as a 64-bit value (zero-extended to 64 from `size`).
// `size` is the instruction's `op_size`.
inline bool read_operand(Cpu& cpu, const Insn& insn, const Operand& op, u8 size, u64& out) noexcept {
    switch (op.kind) {
        case OperandKind::None:
            out = 0;
            return true;

        case OperandKind::Reg: {
            // High-byte aliases (AH/CH/DH/BH at indices 16..19) need r8()'s
            // special path. For all other sizes / indices, cpu.r() + mask.
            if (size == 1 && op.reg >= 16 && op.reg < 20) {
                out = static_cast<u64>(cpu.r8(op.reg));
                return true;
            }
            const u64 raw = cpu.r(op.reg);
            out = raw & mask_for(size);
            return true;
        }

        case OperandKind::Imm:
            out = static_cast<u64>(op.imm) & mask_for(size);
            return true;

        case OperandKind::Rel:
            // For control flow only; not read as a value.
            out = static_cast<u64>(op.imm);
            return true;

        case OperandKind::Mem:
        case OperandKind::RipRelMem: {
            auto* mp = cpu.mem_read();
            if (mp == nullptr) {
                cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure,
                              "no read provider");
                return false;
            }
            const GuestAddr addr = ea(cpu, insn, op);

            // DR data-access breakpoint (load).
            if (cpu.data_breakpoint_at(addr, /*is_write*/false, size)) {
                cpu.set_fault(FaultKind::Breakpoint, addr, Status::Ok, "DR data load breakpoint");
                return false;
            }

            // Fire MemRead hooks (no return value -- to abort, hook calls cpu.halt()).
            if (auto* hm = cpu.hooks(); hm != nullptr && hm->any(HookKind::MemRead)) {
                hm->for_each(HookKind::MemRead, [&](const Hook& h) {
                    if (hook_in_range(h, addr)) {
                        reinterpret_cast<HookMemCb>(h.callback)(h.user_data, cpu, addr, size, 0, false);
                    }
                });
            }

            u8 buf[8] = {};
            Status s = mp->read(addr, size, buf);
            if (fail(s)) {
                cpu.set_fault(FaultKind::PageFault, addr, s, "operand read");
                return false;
            }
            u64 v = 0;
            for (int i = 0; i < size; ++i) v |= (u64{buf[i]} << (8 * i));
            out = v;
            return true;
        }
    }
    return false;
}

// Write a 64-bit value to an operand (low `size` bytes only). Handles the
// x86-64 rule that writing the 32-bit alias of a GPR zero-extends to 64.
inline bool write_operand(Cpu& cpu, const Insn& insn, const Operand& op, u8 size, u64 value) noexcept {
    switch (op.kind) {
        case OperandKind::Reg: {
            switch (size) {
                case 1: cpu.set_r8 (op.reg, static_cast<u8 >(value)); return true;
                case 2: cpu.set_r16(op.reg, static_cast<u16>(value)); return true;
                case 4: cpu.set_r32(op.reg, static_cast<u32>(value)); return true;  // zero-ext to 64
                case 8: cpu.set_r64(op.reg, value); return true;
                default:
                    cpu.set_fault(FaultKind::InternalError, cpu.rip(), Status::Internal,
                                  "bad reg width");
                    return false;
            }
        }

        case OperandKind::Mem:
        case OperandKind::RipRelMem: {
            auto* mp = cpu.mem_write();
            if (mp == nullptr) {
                cpu.set_fault(FaultKind::ProviderFailure, ea(cpu, insn, op),
                              Status::ProviderFailure, "no write provider");
                return false;
            }
            const GuestAddr addr = ea(cpu, insn, op);

            // DR data-access breakpoint (store).
            if (cpu.data_breakpoint_at(addr, /*is_write*/true, size)) {
                cpu.set_fault(FaultKind::Breakpoint, addr, Status::Ok, "DR data store breakpoint");
                return false;
            }

            // Fire MemWrite hooks before the write lands.
            if (auto* hm = cpu.hooks(); hm != nullptr && hm->any(HookKind::MemWrite)) {
                hm->for_each(HookKind::MemWrite, [&](const Hook& h) {
                    if (hook_in_range(h, addr)) {
                        reinterpret_cast<HookMemCb>(h.callback)(h.user_data, cpu, addr, size, value, true);
                    }
                });
            }

            u8 buf[8] = {};
            for (int i = 0; i < size; ++i) buf[i] = static_cast<u8>((value >> (8 * i)) & 0xFFu);
            Status s = mp->write(addr, size, buf);
            if (fail(s)) {
                cpu.set_fault(kind_from_status(s), addr, s, "operand write");
                return false;
            }
            return true;
        }

        case OperandKind::None:
        case OperandKind::Imm:
        case OperandKind::Rel:
            cpu.set_fault(FaultKind::InternalError, cpu.rip(), Status::Internal,
                          "write to non-writable operand kind");
            return false;
    }
    return false;
}

// Sign-extend an N-byte value to 64 bits.
inline i64 sign_extend(u64 v, u8 size) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    if (v & sb) return static_cast<i64>((v & m) | ~m);
    return static_cast<i64>(v & m);
}

// Mask of "shift amount" bits per operand size: 5 bits for 32/16/8-bit ops,
// 6 bits for 64-bit ops. (Intel SDM Vol 2A, SHL/SHR/SAR.)
inline u8 shift_count_mask(u8 size) noexcept {
    return size == 8 ? 0x3F : 0x1F;
}

} // namespace emu::handlers
