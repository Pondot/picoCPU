// Emulator: FakeMemoryProvider for unit-test backing.
//
// Keeps a sorted vector of (base, bytes) regions. Reads walk regions
// in-order. Non-overlap is enforced at add time. Not designed for hot paths.

#include "emu/memory.h"

#include <algorithm>
#include <cstring>

namespace emu {

namespace {

// Inclusive overlap: [a, a+la) vs [b, b+lb).
bool ranges_overlap(GuestAddr a, usize la, GuestAddr b, usize lb) noexcept {
    return !(a + la <= b || b + lb <= a);
}

} // namespace

Status FakeMemoryProvider::add_region(GuestAddr base, std::vector<u8> bytes) noexcept {
    if (bytes.empty()) return Status::InvalidArgument;
    for (const auto& r : regions_) {
        if (ranges_overlap(base, bytes.size(), r.base, r.bytes.size())) {
            return Status::InvalidArgument;
        }
    }
    regions_.push_back(Region{base, std::move(bytes)});
    std::sort(regions_.begin(), regions_.end(),
              [](const Region& a, const Region& b) { return a.base < b.base; });
    return Status::Ok;
}

Status FakeMemoryProvider::read(GuestAddr addr, usize size, void* out) noexcept {
    if (size == 0) return Status::Ok;
    if (out == nullptr) return Status::InvalidArgument;

    auto* dst = static_cast<u8*>(out);
    GuestAddr cur = addr;
    usize remaining = size;

    while (remaining > 0) {
        const Region* hit = nullptr;
        usize hit_off = 0;
        for (const auto& r : regions_) {
            if (cur >= r.base && cur < r.base + r.bytes.size()) {
                hit = &r;
                hit_off = static_cast<usize>(cur - r.base);
                break;
            }
        }
        if (hit == nullptr) return Status::UnmappedRead;

        usize avail = hit->bytes.size() - hit_off;
        usize n = remaining < avail ? remaining : avail;
        std::memcpy(dst, hit->bytes.data() + hit_off, n);
        dst       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

bool FakeMemoryProvider::is_mapped(GuestAddr addr) const noexcept {
    for (const auto& r : regions_) {
        if (addr >= r.base && addr < r.base + r.bytes.size()) return true;
    }
    return false;
}

} // namespace emu
