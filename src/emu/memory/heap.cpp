// Heap implementation.

#include "emu/heap.h"

#include "emu/logger.h"

#include <cstring>

namespace emu {

namespace {
inline u64 round_up(u64 v, u64 to) noexcept { return (v + (to - 1)) & ~(to - 1); }
} // namespace

u32 Heap::class_for(u64 bytes) noexcept {
    u64 sz = round_up(bytes < ALIGN ? ALIGN : bytes, ALIGN);
    u32 cls = 0;
    u64 cap = ALIGN;
    while (cap < sz && cls < SIZE_CLASSES - 1) {
        cap <<= 1;
        ++cls;
    }
    return cls;
}

Heap::Heap(ShadowPages& shadow, GuestAddr base, u64 size) noexcept
    : shadow_(&shadow), base_(base), size_(size), bump_(0) {
    shadow.add_private_region(base, size);
}

GuestAddr Heap::alloc(u64 bytes) noexcept {
    if (bytes == 0) return 0;
    const u32 cls = class_for(bytes);
    const u64 block_sz = ALIGN << cls;

    // Pull from freelist if available. Freelist nodes store the next pointer
    // as the first 8 bytes inside the freed block (in guest memory).
    if (freelist_[cls] != 0) {
        const GuestAddr head = freelist_[cls];
        u64 next = 0;
        Status s = shadow_->read(head, sizeof(next), &next);
        if (fail(s)) {
            LOG_WARN("Heap: freelist read failed at 0x%llx", (unsigned long long)head);
            freelist_[cls] = 0;
        } else {
            freelist_[cls] = static_cast<GuestAddr>(next);
            return head;
        }
    }

    // Bump.
    if (bump_ + block_sz > size_) {
        LOG_WARN("Heap: OOM (bump=%llu need=%llu cap=%llu)",
                 (unsigned long long)bump_, (unsigned long long)block_sz,
                 (unsigned long long)size_);
        return 0;
    }
    const GuestAddr out = base_ + bump_;
    bump_ += block_sz;
    return out;
}

void Heap::free(GuestAddr addr, u64 bytes) noexcept {
    if (addr == 0 || bytes == 0) return;
    if (addr < base_ || addr >= base_ + size_) return;     // not from us -- silently ignore
    const u32 cls = class_for(bytes);
    const u64 next = static_cast<u64>(freelist_[cls]);
    (void)shadow_->write(addr, sizeof(next), &next);
    freelist_[cls] = addr;
}

} // namespace emu
