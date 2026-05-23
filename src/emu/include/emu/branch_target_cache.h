// Branch target cache.
//
// Maps `(branch instruction PC) -> (last observed target PC)`. Direct-mapped,
// power-of-two size, no eviction policy beyond the natural overwrite on
// index collision.
//
// For direct branches the target is constant -- the cache just lets the
// dispatcher skip the BlockCache hash on hot paths. For indirect
// branches/RET the cache predicts the most-recent target; the dispatcher
// always verifies the actual computed RIP matches before trusting it.

#pragma once

#include "emu/types.h"

#include <array>

namespace emu {

class BranchTargetCache {
public:
    static constexpr u32 ENTRIES = 256;        // direct-mapped power of two

    [[nodiscard]] GuestAddr predict(GuestAddr branch_pc) const noexcept {
        const u32 idx = static_cast<u32>((branch_pc >> 2) & (ENTRIES - 1));
        const auto& e = entries_[idx];
        if (e.branch_pc == branch_pc) return e.target_pc;
        return 0;
    }

    void record(GuestAddr branch_pc, GuestAddr target_pc) noexcept {
        const u32 idx = static_cast<u32>((branch_pc >> 2) & (ENTRIES - 1));
        entries_[idx] = {branch_pc, target_pc};
    }

    void clear() noexcept {
        for (auto& e : entries_) e = {};
        hits_ = misses_ = 0;
    }

    [[nodiscard]] u64 hits()   const noexcept { return hits_; }
    [[nodiscard]] u64 misses() const noexcept { return misses_; }
    void              count_hit()  noexcept { ++hits_; }
    void              count_miss() noexcept { ++misses_; }

private:
    struct Entry { GuestAddr branch_pc = 0; GuestAddr target_pc = 0; };
    std::array<Entry, ENTRIES> entries_{};
    u64 hits_   = 0;
    u64 misses_ = 0;
};

} // namespace emu
