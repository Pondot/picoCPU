// Three-tier read cache over a MemoryProvider chain.
//
// Layout:
//   L1: 64 B cache lines, direct-mapped, 64 entries.    Hot stack/local reads.
//   L2: 4 KiB pages, direct-mapped, 256 entries.        Function body + heap.
//   L3: 2 MiB regions, LRU, 16 entries.                 Bulk RPM amortization.
//
// Writes go straight through to the next layer; each layer also invalidates
// the affected cached entries above the write. The chain is:
//
//     handler -> L1 -> L2 -> L3 -> ShadowPages -> RPM
//
// In Phase 3 we plumb the chain end-to-end; the real perf win shows up once
// Phase 4a engages the block cache so the same chain is reused across
// instructions of the same function.

#pragma once

#include "emu/error.h"
#include "emu/memory.h"
#include "emu/types.h"

#include <array>
#include <cstring>
#include <vector>

namespace emu {

class L1Cache final : public MemoryProvider {
public:
    explicit L1Cache(MemoryProvider* next) noexcept : next_(next) {}

    [[nodiscard]] Status read (GuestAddr addr, usize size, void* out)        noexcept override;
    [[nodiscard]] Status write(GuestAddr addr, usize size, const void* data) noexcept override;
    void invalidate(GuestAddr addr, usize size) noexcept override;
    [[nodiscard]] bool is_mapped(GuestAddr a) const noexcept override { return next_ && next_->is_mapped(a); }

    // Diagnostics.
    [[nodiscard]] u64 hits()   const noexcept { return hits_;   }
    [[nodiscard]] u64 misses() const noexcept { return misses_; }
    void               reset_stats() noexcept { hits_ = misses_ = 0; }
    void               flush() noexcept;

private:
    static constexpr u32 ENTRIES = 64;
    struct Line {
        u64  tag = 0;       // line base address (addr & ~63)
        bool valid = false;
        u8   data[64]{};
    };
    MemoryProvider* next_;
    std::array<Line, ENTRIES> lines_{};
    u64 hits_   = 0;
    u64 misses_ = 0;
};

class L2Cache final : public MemoryProvider {
public:
    explicit L2Cache(MemoryProvider* next) noexcept : next_(next) {}

    [[nodiscard]] Status read (GuestAddr addr, usize size, void* out)        noexcept override;
    [[nodiscard]] Status write(GuestAddr addr, usize size, const void* data) noexcept override;
    void invalidate(GuestAddr addr, usize size) noexcept override;
    [[nodiscard]] bool is_mapped(GuestAddr a) const noexcept override { return next_ && next_->is_mapped(a); }

    [[nodiscard]] u64 hits()   const noexcept { return hits_;   }
    [[nodiscard]] u64 misses() const noexcept { return misses_; }
    void               reset_stats() noexcept { hits_ = misses_ = 0; }
    void               flush() noexcept;

private:
    static constexpr u32 ENTRIES = 256;
    struct Page {
        u64  tag = 0;
        bool valid = false;
        u8   data[4096]{};
    };
    MemoryProvider* next_;
    // Page is large (4 KiB) so we heap-allocate the table on first use.
    std::vector<Page> pages_;
    u64 hits_   = 0;
    u64 misses_ = 0;
};

class L3Cache final : public MemoryProvider {
public:
    explicit L3Cache(MemoryProvider* next) noexcept : next_(next) {}

    [[nodiscard]] Status read (GuestAddr addr, usize size, void* out)        noexcept override;
    [[nodiscard]] Status write(GuestAddr addr, usize size, const void* data) noexcept override;
    void invalidate(GuestAddr addr, usize size) noexcept override;
    [[nodiscard]] bool is_mapped(GuestAddr a) const noexcept override { return next_ && next_->is_mapped(a); }

    // Eagerly prefetch a 2 MiB region containing `addr` (one bulk read into L3).
    [[nodiscard]] Status prefetch_region(GuestAddr addr) noexcept;

    [[nodiscard]] u64 hits()   const noexcept { return hits_;   }
    [[nodiscard]] u64 misses() const noexcept { return misses_; }
    [[nodiscard]] u64 fetches() const noexcept { return fetches_; }
    void               reset_stats() noexcept { hits_ = misses_ = fetches_ = 0; }
    void               flush() noexcept;

private:
    static constexpr u32 ENTRIES = 16;
    struct Region {
        u64                base = 0;
        bool               valid = false;
        std::vector<u8>    data;       // 2 MiB on first use
        u64                last_use = 0;
    };

    MemoryProvider* next_;
    std::array<Region, ENTRIES> regions_{};
    u64 use_counter_ = 0;
    u64 hits_    = 0;
    u64 misses_  = 0;
    u64 fetches_ = 0;   // calls to next_->read for region fills (the perf lever)
};

} // namespace emu
