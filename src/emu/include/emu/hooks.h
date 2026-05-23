// Hook system.
//
// One hook list per kind; each hook carries a callback, user_data, and an
// inclusive [begin, end] address range filter (or "any" when begin > end).
// The dispatcher checks `if (!hooks_of(kind).empty())` before iterating, so
// uninstrumented runs pay zero cost.
//
// Hook kinds:
//   Block      -- fired before a basic block executes (one call per block).
//   Code       -- fired before each individual instruction.
//   MemRead    -- fired before a memory read (operand or LEA-derived address).
//   MemWrite   -- fired before a memory write.
//   Insn       -- fired before a specific instruction kind (CPUID, RDTSC, ...).
//   Fault      -- fired when the dispatcher latches a fault (replaces the
//                cpu-level single-fault hook on installation).
//
// Callbacks for Block/Code/Mem return void; their return value cannot abort
// emulation. To stop, the callback calls `Cpu::halt()`.

#pragma once

#include "emu/cpu.h"
#include "emu/fault.h"
#include "emu/ir.h"
#include "emu/types.h"

#include <vector>

namespace emu {

enum class HookKind : u8 {
    Block = 0,
    Code,
    MemRead,
    MemWrite,
    Insn,        // dispatch on OpKind via filter_op
    Fault,
    _Count,
};

using HookCodeCb  = void (*)(void* user, Cpu& cpu, GuestAddr rip, u32 size);
using HookBlockCb = void (*)(void* user, Cpu& cpu, GuestAddr block_pc, u32 byte_size);
using HookMemCb   = void (*)(void* user, Cpu& cpu, GuestAddr addr, u32 size,
                              u64 value /*for writes*/, bool is_write);
using HookInsnCb  = void (*)(void* user, Cpu& cpu, const Insn& insn);
using HookFaultCb = void (*)(void* user, Cpu& cpu, const Fault& fault);

struct Hook {
    HookKind  kind;
    void*     callback;       // erased -- caller knows the right type for `kind`
    void*     user_data;
    GuestAddr begin;
    GuestAddr end;
    OpKind    filter_op;      // for HookKind::Insn -- matches insn.kind; OpKind::Invalid = any
};

class HookManager {
public:
    using HookId = u32;
    static constexpr HookId INVALID_ID = 0xFFFF'FFFFu;

    HookId add(const Hook& h) noexcept {
        const u32 idx = static_cast<u32>(static_cast<u8>(h.kind));
        auto& vec = lists_[idx];
        const HookId id = next_id_++;
        vec.push_back({id, h});
        return id;
    }

    void remove(HookId id) noexcept {
        for (auto& list : lists_) {
            for (auto it = list.begin(); it != list.end(); ++it) {
                if (it->id == id) { list.erase(it); return; }
            }
        }
    }

    void clear() noexcept {
        for (auto& l : lists_) l.clear();
    }

    [[nodiscard]] bool any(HookKind k) const noexcept {
        return !lists_[static_cast<u8>(k)].empty();
    }

    template <typename Fn>
    void for_each(HookKind k, Fn&& fn) const {
        const auto& list = lists_[static_cast<u8>(k)];
        for (const auto& entry : list) fn(entry.h);
    }

private:
    struct Entry {
        HookId id;
        Hook   h;
    };
    std::vector<Entry> lists_[static_cast<u8>(HookKind::_Count)];
    HookId             next_id_ = 1;
};

// Range filter helper: returns true if `addr` is inside [begin, end], or if
// begin > end (the "any address" wildcard).
inline bool hook_in_range(const Hook& h, GuestAddr addr) noexcept {
    if (h.begin > h.end) return true;
    return addr >= h.begin && addr <= h.end;
}

} // namespace emu
