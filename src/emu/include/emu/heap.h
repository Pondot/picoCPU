// Guest-space heap allocator.
//
// Owns a virtual address range inside ShadowPages and hands out aligned
// blocks. Bump allocator first; freed blocks return to a singly-linked
// freelist keyed by size class. No coalescing yet (a 16-class quick-fit
// allocator is plenty for the workloads we'll throw at Phase 3-4 targets).
//
// The Heap does NOT own pages directly -- it asks ShadowPages to materialize
// pages on first access. Writes through the cache chain land in shadow.

#pragma once

#include "emu/error.h"
#include "emu/shadow_pages.h"
#include "emu/types.h"

#include <array>

namespace emu {

class Heap {
public:
    // base+size must be page-aligned (4 KiB). Heap registers `[base, base+size)`
    // as a private region with `shadow` so reads-before-write return zero.
    Heap(ShadowPages& shadow, GuestAddr base, u64 size) noexcept;

    // Allocate a block of at least `bytes`, aligned to 16. Returns 0 on OOM.
    [[nodiscard]] GuestAddr alloc(u64 bytes) noexcept;

    // Free a previously-allocated block. Size must match what alloc returned.
    void free(GuestAddr addr, u64 bytes) noexcept;

    [[nodiscard]] GuestAddr base() const noexcept { return base_; }
    [[nodiscard]] u64       size() const noexcept { return size_; }
    [[nodiscard]] u64       bump_offset() const noexcept { return bump_; }

private:
    static constexpr u64 ALIGN = 16;
    static constexpr u32 SIZE_CLASSES = 16;
    static u32 class_for(u64 bytes) noexcept;   // -> class index, rounded up

    ShadowPages* shadow_;
    GuestAddr    base_;
    u64          size_;
    u64          bump_;                          // offset of next bump
    std::array<GuestAddr, SIZE_CLASSES> freelist_{};   // head per class (0 = empty)
};

} // namespace emu
