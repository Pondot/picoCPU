// IAT stub layer -- implementation.

#include "emu/iat_stubs.h"
#include "emu/cpu.h"
#include "emu/logger.h"

#include <cctype>
#include <cstring>

namespace emu {

namespace {

std::string make_key(const std::string& dll, const std::string& name) noexcept {
    std::string k;
    k.reserve(dll.size() + 1 + name.size());
    for (char c : dll) k.push_back(static_cast<char>(std::tolower(c)));
    k.push_back('|');
    k.append(name);
    return k;
}

} // namespace

void IatStubs::install(ShadowPages& shadow) noexcept {
    shadow.add_private_region(kStubBase, kStubSize);
    // Mark the region as executable code: it isn't really code (we never
    // decode bytes there), but the dispatcher uses RIP-in-region as the
    // signal to call us. The "in_region" check happens before decode.
    (void)shadow;
}

GuestAddr IatStubs::bind(ShadowPages& shadow, GuestAddr iat_addr,
                         std::string dll, std::string name) noexcept {
    const std::size_t idx = entries_.size();
    const GuestAddr stub_addr = kStubBase + idx * kStubStride;
    if (stub_addr + kStubStride > kStubBase + kStubSize) {
        LOG_WARN("iat_stubs: region full (%zu entries)", idx);
        return 0;
    }
    entries_.push_back(Entry{std::move(dll), std::move(name)});

    // Patch the IAT entry to point at the stub.
    u8 buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<u8>((stub_addr >> (8 * i)) & 0xFFu);
    if (Status s = shadow.write(iat_addr, 8, buf); fail(s)) {
        LOG_WARN("iat_stubs: patching IAT@0x%llx failed",
                 (unsigned long long)iat_addr);
    }
    return stub_addr;
}

void IatStubs::set_stub(const char* dll, const char* name, StubFn fn) noexcept {
    stubs_[make_key(dll ? dll : "", name ? name : "")] = fn;
}

bool IatStubs::dispatch(Cpu& cpu, GuestAddr rip) noexcept {
    if (rip < kStubBase || rip >= kStubBase + kStubSize) return false;
    const std::size_t idx = static_cast<std::size_t>((rip - kStubBase) / kStubStride);
    if (idx >= entries_.size()) {
        cpu.set_fault(FaultKind::InvalidOpcode, rip, Status::UnsupportedOpcode,
                      "iat stub idx out of range");
        return false;
    }
    const Entry& e = entries_[idx];

    // Look up stub by (dll, name) first; fall back to ("", name) for stubs
    // registered without a specific DLL (handles api-ms-* redirector flavors
    // pointing at the same logical function); else use the default.
    StubFn fn = nullptr;
    if (auto it = stubs_.find(make_key(e.dll, e.name)); it != stubs_.end()) fn = it->second;
    else if (auto it2 = stubs_.find(make_key("", e.name)); it2 != stubs_.end()) fn = it2->second;
    else fn = default_stub_;

    if (fn) {
        fn(cpu, e.dll.c_str(), e.name.c_str());
    } else {
        LOG_WARN("iat_stubs: unhandled call to %s!%s, defaulting RAX=0",
                 e.dll.c_str(), e.name.c_str());
        cpu.set_r64(reg::RAX, 0);
    }

    // Synthesize RET: pop return address from [RSP], add 8 to RSP, set RIP.
    auto* mp = cpu.mem_read();
    if (!mp) {
        cpu.set_fault(FaultKind::ProviderFailure, cpu.r(reg::RSP),
                      Status::ProviderFailure, "iat stub: no read provider");
        return false;
    }
    u8 buf[8] = {};
    const GuestAddr rsp = cpu.r(reg::RSP);
    if (Status s = mp->read(rsp, 8, buf); fail(s)) {
        cpu.set_fault(FaultKind::PageFault, rsp, s, "iat stub: ret read");
        return false;
    }
    u64 ret_addr = 0;
    for (int i = 0; i < 8; ++i) ret_addr |= (u64{buf[i]} << (8 * i));
    cpu.set_r64(reg::RSP, rsp + 8);
    cpu.set_rip(ret_addr);
    return true;
}

} // namespace emu
