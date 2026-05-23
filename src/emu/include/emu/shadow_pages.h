// ShadowPages -- 4 KiB copy-on-write writable provider.
//
// Backed by an optional read-only `backing` provider (e.g. RPM into a target
// process). Reads consult shadow first, then fall back to backing. Writes
// always go to shadow, copying the page in from backing on first touch.
//
// Independent regions can be reserved as "private" -- bytes that don't exist
// in any backing source. Reads to a private region prior to any write return
// zero-filled bytes (we lazily create a zero-filled shadow page on read).

#pragma once

#include "emu/error.h"
#include "emu/memory.h"
#include "emu/types.h"

#include <array>
#include <unordered_map>
#include <vector>

namespace emu {

class ShadowPages final : public MemoryProvider {
public:
    explicit ShadowPages(MemoryProvider* backing = nullptr) noexcept : backing_(backing) {}

    [[nodiscard]] Status read (GuestAddr addr, usize size, void* out)        noexcept override;
    [[nodiscard]] Status write(GuestAddr addr, usize size, const void* data) noexcept override;
    [[nodiscard]] bool   is_mapped(GuestAddr addr) const noexcept override;

    // Reserve a private (no-backing) range. Reads before any write return
    // zeros; writes persist into shadow. Used for guest stack/heap.
    void add_private_region(GuestAddr base, u64 size) noexcept;

    // Per-region permissions. Default is RW for private regions and R-only
    // for backed (RPM) regions. Calling protect() narrows or widens access
    // for a range. Bit 0=R, 1=W, 2=X.
    static constexpr u8 PROT_R = 1u << 0;
    static constexpr u8 PROT_W = 1u << 1;
    static constexpr u8 PROT_X = 1u << 2;
    void protect(GuestAddr base, u64 size, u8 prot) noexcept;
    [[nodiscard]] u8 perm_of(GuestAddr addr) const noexcept;

    // Mark a range as containing executable code. Writes that overlap a code
    // range fire the write listener -- used for SMC: BlockCache invalidates
    // any block whose PC overlaps a written byte.
    void add_code_range(GuestAddr base, u64 size) noexcept;

    // Listener invoked on writes that overlap any registered code range.
    // Signature: void(*)(void* user, GuestAddr addr, usize size).
    using WriteListener = void (*)(void* user, GuestAddr addr, usize size);
    void set_write_listener(WriteListener cb, void* user) noexcept {
        write_cb_   = cb;
        write_user_ = user;
    }

    // Diagnostics / tests.
    [[nodiscard]] usize shadow_page_count() const noexcept { return pages_.size(); }

private:
    struct Page {
        std::array<u8, 4096> data{};
        bool initialized = false;   // copied in from backing or zeroed
    };

    Page& get_or_make_page_(GuestAddr page_addr) noexcept;
    bool  range_is_private_(GuestAddr addr, usize size) const noexcept;

    MemoryProvider* backing_ = nullptr;
    std::unordered_map<u64, Page> pages_;

    struct PrivateRange { GuestAddr base; u64 size; };
    std::vector<PrivateRange> private_ranges_;
    std::vector<PrivateRange> code_ranges_;

    struct ProtRange { GuestAddr base; u64 size; u8 prot; };
    std::vector<ProtRange> prot_ranges_;
    WriteListener             write_cb_   = nullptr;
    void*                     write_user_ = nullptr;
};

} // namespace emu
