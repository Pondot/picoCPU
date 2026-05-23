// ShadowPages implementation.

#include "emu/shadow_pages.h"

#include "emu/logger.h"

#include <cstring>

namespace emu {

namespace {
constexpr u64 PAGE_BASE_MASK = ~u64{0xFFF};
}

void ShadowPages::add_private_region(GuestAddr base, u64 size) noexcept {
    private_ranges_.push_back(PrivateRange{base, size});
}

void ShadowPages::add_code_range(GuestAddr base, u64 size) noexcept {
    code_ranges_.push_back(PrivateRange{base, size});
}

void ShadowPages::protect(GuestAddr base, u64 size, u8 prot) noexcept {
    prot_ranges_.push_back(ProtRange{base, size, prot});
}

u8 ShadowPages::perm_of(GuestAddr addr) const noexcept {
    // Most-recent matching range wins.
    for (auto it = prot_ranges_.rbegin(); it != prot_ranges_.rend(); ++it) {
        if (addr >= it->base && addr < it->base + it->size) return it->prot;
    }
    // Default: private regions are RW; backed regions are R only.
    if (range_is_private_(addr, 1)) return PROT_R | PROT_W;
    return PROT_R;
}

bool ShadowPages::range_is_private_(GuestAddr addr, usize size) const noexcept {
    const GuestAddr lo = addr;
    const GuestAddr hi = addr + size;
    for (const auto& r : private_ranges_) {
        const GuestAddr r_hi = r.base + r.size;
        if (lo >= r.base && hi <= r_hi) return true;
    }
    return false;
}

ShadowPages::Page& ShadowPages::get_or_make_page_(GuestAddr page_addr) noexcept {
    auto [it, inserted] = pages_.try_emplace(page_addr);
    Page& page = it->second;
    if (inserted) {
        // Initialize from backing if available and the region is not private.
        if (!range_is_private_(page_addr, 4096) && backing_ != nullptr) {
            Status s = backing_->read(page_addr, 4096, page.data.data());
            if (fail(s)) {
                // Couldn't pull from backing -- leave zero-initialized.
                LOG_DEBUG("ShadowPages: backing read for page 0x%llx failed (%.*s); zero-fill",
                          (unsigned long long)page_addr,
                          static_cast<int>(to_string(s).size()),
                          to_string(s).data());
            }
        }
        page.initialized = true;
    }
    return page;
}

Status ShadowPages::read(GuestAddr addr, usize size, void* out) noexcept {
    if (size == 0)        return Status::Ok;
    if (out == nullptr)   return Status::InvalidArgument;

    auto* dst = static_cast<u8*>(out);

    GuestAddr cur = addr;
    usize     remaining = size;
    while (remaining > 0) {
        const GuestAddr page_addr = cur & PAGE_BASE_MASK;
        const u32  page_off  = static_cast<u32>(cur - page_addr);
        const usize n        = (4096 - page_off) < remaining ? (4096 - page_off) : remaining;

        auto it = pages_.find(page_addr);
        if (it != pages_.end()) {
            std::memcpy(dst, it->second.data.data() + page_off, n);
        } else if (range_is_private_(page_addr, 4096)) {
            // Private region not yet touched -> zeroes.
            std::memset(dst, 0, n);
        } else if (backing_ != nullptr) {
            // Pass through to backing without instantiating a shadow page.
            if (Status s = backing_->read(cur, n, dst); fail(s)) return s;
        } else {
            return Status::UnmappedRead;
        }
        dst       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

Status ShadowPages::write(GuestAddr addr, usize size, const void* data) noexcept {
    if (size == 0)         return Status::Ok;
    if (data == nullptr)   return Status::InvalidArgument;

    // Enforce per-region W permission, but only when the user has explicitly
    // installed prot ranges. Default (no prot ranges) is permissive to keep
    // existing tests/targets working.
    if (!prot_ranges_.empty()) {
        const GuestAddr end = addr + size;
        for (GuestAddr cur = addr; cur < end; ) {
            const u8 p = perm_of(cur);
            if (!(p & PROT_W)) return Status::ProtectionViolation;
            // step forward by 1 byte (slow, but only triggers if prot ranges
            // are configured; tests opt in).
            ++cur;
        }
    }

    const auto* src = static_cast<const u8*>(data);

    GuestAddr cur = addr;
    usize     remaining = size;
    while (remaining > 0) {
        const GuestAddr page_addr = cur & PAGE_BASE_MASK;
        const u32  page_off  = static_cast<u32>(cur - page_addr);
        const usize n        = (4096 - page_off) < remaining ? (4096 - page_off) : remaining;

        Page& page = get_or_make_page_(page_addr);
        std::memcpy(page.data.data() + page_off, src, n);

        src       += n;
        cur       += n;
        remaining -= n;
    }

    // SMC: if the write overlaps any registered code range, notify the listener.
    if (write_cb_ != nullptr) {
        const GuestAddr lo = addr;
        const GuestAddr hi = addr + size;
        for (const auto& r : code_ranges_) {
            const GuestAddr r_hi = r.base + r.size;
            if (lo < r_hi && hi > r.base) {
                write_cb_(write_user_, addr, size);
                break;
            }
        }
    }
    return Status::Ok;
}

bool ShadowPages::is_mapped(GuestAddr addr) const noexcept {
    const GuestAddr page_addr = addr & PAGE_BASE_MASK;
    if (pages_.find(page_addr) != pages_.end()) return true;
    if (range_is_private_(addr, 1)) return true;
    if (backing_ != nullptr) return backing_->is_mapped(addr);
    return false;
}

} // namespace emu
