// L3Cache: 2 MiB LRU regions, 16 entries.
//
// This is the actual RPM-amortization lever. One 2 MiB ReadProcessMemory
// call buys ~32K cache lines of guaranteed-fast access from L1/L2.

#include "emu/cache.h"

#include "emu/logger.h"

#include <cstring>

namespace emu {

namespace {
constexpr u32 REGION_SHIFT = 21;             // 2 MiB
constexpr u64 REGION_BYTES = u64{1} << REGION_SHIFT;
constexpr u64 REGION_MASK  = ~(REGION_BYTES - 1);
}

void L3Cache::flush() noexcept {
    for (auto& r : regions_) { r.valid = false; r.base = 0; r.data.clear(); }
    use_counter_ = 0;
}

void L3Cache::invalidate(GuestAddr addr, usize size) noexcept {
    if (size == 0) return;
    const GuestAddr start = addr & REGION_MASK;
    const GuestAddr end   = (addr + size - 1) & REGION_MASK;
    for (GuestAddr r = start; ; r += REGION_BYTES) {
        for (auto& slot : regions_) {
            if (slot.valid && slot.base == r) {
                slot.valid = false;
                slot.data.clear();
            }
        }
        if (r == end) break;
    }
}

Status L3Cache::prefetch_region(GuestAddr addr) noexcept {
    // Find or create the slot, then refill it from next_.
    const GuestAddr base = addr & REGION_MASK;

    // Look for an existing entry -- refresh if found.
    for (auto& slot : regions_) {
        if (slot.valid && slot.base == base) {
            slot.last_use = ++use_counter_;
            return Status::Ok;
        }
    }
    // Otherwise pick an LRU slot.
    Region* victim = &regions_[0];
    for (auto& slot : regions_) {
        if (!slot.valid)               { victim = &slot; break; }
        if (slot.last_use < victim->last_use) victim = &slot;
    }
    if (victim->data.size() != REGION_BYTES) victim->data.assign(REGION_BYTES, 0);
    if (next_ == nullptr) return Status::ProviderFailure;

    ++fetches_;
    if (Status s = next_->read(base, REGION_BYTES, victim->data.data()); fail(s)) {
        victim->valid = false;
        victim->data.clear();
        LOG_DEBUG("L3 prefetch failed at 0x%llx (%.*s)",
                  (unsigned long long)base,
                  static_cast<int>(to_string(s).size()),
                  to_string(s).data());
        return s;
    }
    victim->base = base;
    victim->valid = true;
    victim->last_use = ++use_counter_;
    return Status::Ok;
}

Status L3Cache::read(GuestAddr addr, usize size, void* out) noexcept {
    if (size == 0) return Status::Ok;
    if (next_ == nullptr) return Status::ProviderFailure;

    auto* dst = static_cast<u8*>(out);

    GuestAddr cur = addr;
    usize remaining = size;
    while (remaining > 0) {
        const GuestAddr region_base = cur & REGION_MASK;
        const u64       off_in_region = cur - region_base;
        const usize     n = (REGION_BYTES - off_in_region) < remaining
                          ? static_cast<usize>(REGION_BYTES - off_in_region)
                          : remaining;

        // Look up.
        Region* hit = nullptr;
        for (auto& slot : regions_) {
            if (slot.valid && slot.base == region_base) { hit = &slot; break; }
        }
        if (hit != nullptr) {
            ++hits_;
            hit->last_use = ++use_counter_;
            std::memcpy(dst, hit->data.data() + off_in_region, n);
        } else {
            ++misses_;
            // Try a bulk fetch first; if that fails (e.g. the 2 MiB region
            // straddles unmapped memory in the target process), fall back to
            // a direct passthrough read for this request only -- no caching.
            if (fail(prefetch_region(region_base))) {
                if (Status s = next_->read(cur, n, dst); fail(s)) return s;
            } else {
                for (auto& slot : regions_) {
                    if (slot.valid && slot.base == region_base) {
                        std::memcpy(dst, slot.data.data() + off_in_region, n);
                        break;
                    }
                }
            }
        }
        dst       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

Status L3Cache::write(GuestAddr addr, usize size, const void* data) noexcept {
    if (size == 0) return Status::Ok;
    if (next_ == nullptr) return Status::ProviderFailure;
    if (Status s = next_->write(addr, size, data); fail(s)) return s;

    const auto* src = static_cast<const u8*>(data);
    GuestAddr cur = addr;
    usize remaining = size;
    while (remaining > 0) {
        const GuestAddr region_base = cur & REGION_MASK;
        const u64       off  = cur - region_base;
        const usize     n    = (REGION_BYTES - off) < remaining
                             ? static_cast<usize>(REGION_BYTES - off)
                             : remaining;
        // In-place update of any matching region; do not invalidate the slot.
        for (auto& slot : regions_) {
            if (slot.valid && slot.base == region_base) {
                std::memcpy(slot.data.data() + off, src, n);
                break;
            }
        }
        src       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

} // namespace emu
