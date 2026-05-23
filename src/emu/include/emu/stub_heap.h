// Tiny first-fit free-list heap used by the IAT HeapAlloc/HeapFree stubs.
//
// Lives in a contiguous guest-VA region (added as a private region in
// ShadowPages by the caller). Tracks allocations in a host-side block list.
// Coalesces adjacent free blocks on free. Not thread-safe; the emulator is
// single-threaded.
//
// Headers are *host-side* -- we don't write any metadata into guest memory.
// That keeps the guest view of allocated pointers as plain bytes (which is
// what malloc-style callers expect) and avoids the "pointer ± 8" header
// gymnastics that real heaps do.

#pragma once

#include "emu/types.h"

#include <vector>

namespace emu {

class StubHeap {
public:
    StubHeap() = default;

    // Reserve [base, base+size). Initially the entire region is one free block.
    void install(GuestAddr base, u64 size) noexcept {
        base_ = base;
        size_ = size;
        blocks_.clear();
        blocks_.push_back(Block{base, size, /*free=*/true});
    }

    // First-fit allocation. Round size up to 16 bytes. Returns 0 on OOM.
    GuestAddr alloc(u64 sz) noexcept {
        const u64 aligned = (sz + 15ull) & ~u64{15};
        if (aligned == 0) return 0;
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            Block& b = blocks_[i];
            if (!b.free || b.size < aligned) continue;
            // Split the block: [aligned bytes allocated][remainder free].
            const GuestAddr p = b.addr;
            if (b.size == aligned) {
                b.free = false;
            } else {
                Block tail{b.addr + aligned, b.size - aligned, true};
                b.size = aligned;
                b.free = false;
                blocks_.insert(blocks_.begin() + i + 1, tail);
            }
            return p;
        }
        return 0;
    }

    // Free returns true on success. Coalesces with neighbors.
    bool free(GuestAddr p) noexcept {
        if (p == 0) return true;
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            if (blocks_[i].addr != p) continue;
            if (blocks_[i].free) return false;     // double-free
            blocks_[i].free = true;
            // Coalesce with previous if free.
            if (i > 0 && blocks_[i - 1].free) {
                blocks_[i - 1].size += blocks_[i].size;
                blocks_.erase(blocks_.begin() + i);
                --i;
            }
            // Coalesce with next if free.
            if (i + 1 < blocks_.size() && blocks_[i + 1].free) {
                blocks_[i].size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
            }
            return true;
        }
        return false;     // not found
    }

    // Stats for diagnostics / tests.
    [[nodiscard]] u64 used_bytes() const noexcept {
        u64 t = 0;
        for (const auto& b : blocks_) if (!b.free) t += b.size;
        return t;
    }
    [[nodiscard]] u64 free_bytes() const noexcept {
        u64 t = 0;
        for (const auto& b : blocks_) if (b.free) t += b.size;
        return t;
    }
    [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }

private:
    struct Block {
        GuestAddr addr;
        u64       size;
        bool      free;
    };
    GuestAddr           base_ = 0;
    u64                 size_ = 0;
    std::vector<Block>  blocks_;
};

} // namespace emu
