// L2Cache: 4 KiB pages, direct-mapped, 256 entries (1 MiB total).

#include "emu/cache.h"

#include "emu/logger.h"

#include <cstring>

namespace emu {

namespace {
constexpr u64 PAGE_MASK_ = ~u64{0xFFF};
}

void L2Cache::flush() noexcept {
    for (auto& p : pages_) { p.valid = false; p.tag = 0; }
}

void L2Cache::invalidate(GuestAddr addr, usize size) noexcept {
    if (pages_.empty() || size == 0) return;
    const GuestAddr start = addr & PAGE_MASK_;
    const GuestAddr end   = (addr + size - 1) & PAGE_MASK_;
    for (GuestAddr p = start; ; p += 4096) {
        const u32 idx = static_cast<u32>((p >> 12) & (ENTRIES - 1));
        if (pages_[idx].valid && pages_[idx].tag == p) {
            pages_[idx].valid = false;
        }
        if (p == end) break;
    }
}

Status L2Cache::read(GuestAddr addr, usize size, void* out) noexcept {
    if (size == 0) return Status::Ok;
    if (next_ == nullptr) return Status::ProviderFailure;

    if (pages_.empty()) pages_.resize(ENTRIES);

    auto* dst = static_cast<u8*>(out);

    GuestAddr cur = addr;
    usize remaining = size;
    while (remaining > 0) {
        const GuestAddr page_base = cur & PAGE_MASK_;
        const u32  off    = static_cast<u32>(cur - page_base);
        const usize n     = (4096 - off) < remaining ? (4096 - off) : remaining;
        const u32  idx    = static_cast<u32>((page_base >> 12) & (ENTRIES - 1));
        Page& slot = pages_[idx];

        if (slot.valid && slot.tag == page_base) {
            ++hits_;
        } else {
            ++misses_;
            if (Status s = next_->read(page_base, 4096, slot.data); fail(s)) {
                slot.valid = false;
                return s;
            }
            slot.tag   = page_base;
            slot.valid = true;
        }

        std::memcpy(dst, slot.data + off, n);
        dst       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

Status L2Cache::write(GuestAddr addr, usize size, const void* data) noexcept {
    if (size == 0) return Status::Ok;
    if (next_ == nullptr) return Status::ProviderFailure;
    if (Status s = next_->write(addr, size, data); fail(s)) return s;

    if (pages_.empty()) return Status::Ok;

    const auto* src = static_cast<const u8*>(data);
    GuestAddr cur = addr;
    usize remaining = size;
    while (remaining > 0) {
        const GuestAddr page_base = cur & PAGE_MASK_;
        const u32  off    = static_cast<u32>(cur - page_base);
        const usize n     = (4096 - off) < remaining ? (4096 - off) : remaining;
        const u32  idx    = static_cast<u32>((page_base >> 12) & (ENTRIES - 1));
        Page& slot = pages_[idx];
        if (slot.valid && slot.tag == page_base) {
            std::memcpy(slot.data + off, src, n);
        }
        src       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

} // namespace emu
