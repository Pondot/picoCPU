// Host: Windows process attach + RPM implementation.

#include "win_process.h"

#include "emu/logger.h"

#include <windows.h>
#include <psapi.h>

namespace emu::host {

namespace {

inline HANDLE to_handle(void* p) noexcept { return static_cast<HANDLE>(p); }

} // namespace

Process::~Process() { close_(); }

Process::Process(Process&& other) noexcept
    : handle_(other.handle_), pid_(other.pid_) {
    other.handle_ = nullptr;
    other.pid_    = 0;
}

Process& Process::operator=(Process&& other) noexcept {
    if (this != &other) {
        close_();
        handle_ = other.handle_;
        pid_    = other.pid_;
        other.handle_ = nullptr;
        other.pid_    = 0;
    }
    return *this;
}

void Process::close_() noexcept {
    if (handle_ != nullptr) {
        ::CloseHandle(to_handle(handle_));
        handle_ = nullptr;
    }
    pid_ = 0;
}

Status Process::open(u32 pid) noexcept {
    close_();
    constexpr DWORD desired =
        PROCESS_VM_READ | PROCESS_QUERY_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION;
    HANDLE h = ::OpenProcess(desired, FALSE, pid);
    if (h == nullptr) {
        LOG_ERROR("OpenProcess(pid=%u) failed: GLE=%lu", pid, ::GetLastError());
        return Status::HostOpenFailed;
    }
    handle_ = h;
    pid_    = pid;
    LOG_INFO("Opened pid=%u handle=%p", pid, h);
    return Status::Ok;
}

Status Process::read(GuestAddr addr, usize size, void* out) const noexcept {
    if (!is_open())   return Status::HostOpenFailed;
    if (out == nullptr || size == 0) return Status::InvalidArgument;

    // Iterate in 4 KiB chunks aligned to OS page boundaries. RPM treats a
    // single call that crosses two VirtualAlloc/PE-section regions as a
    // partial read; splitting per page gives us "fully read or fully fail"
    // semantics at the cache-page granularity our L2 expects.
    auto* dst = static_cast<u8*>(out);
    GuestAddr cur = addr;
    usize remaining = size;

    while (remaining > 0) {
        const GuestAddr page_end = (cur & ~u64{0xFFF}) + 0x1000;
        const usize chunk = (page_end - cur) < remaining
                          ? static_cast<usize>(page_end - cur)
                          : remaining;
        SIZE_T got = 0;
        BOOL ok = ::ReadProcessMemory(to_handle(handle_),
                                      reinterpret_cast<LPCVOID>(cur),
                                      dst, chunk, &got);
        if (!ok || got != chunk) {
            LOG_DEBUG("RPM(0x%llx, %llu) -> ok=%d got=%llu GLE=%lu",
                      (unsigned long long)cur, (unsigned long long)chunk,
                      ok ? 1 : 0, (unsigned long long)got,
                      static_cast<unsigned long>(::GetLastError()));
            return Status::HostReadFailed;
        }
        cur       += chunk;
        dst       += chunk;
        remaining -= chunk;
    }
    return Status::Ok;
}

Status Process::query(GuestAddr addr, ProcessRegion& out) const noexcept {
    if (!is_open()) return Status::HostOpenFailed;
    MEMORY_BASIC_INFORMATION mbi{};
    SIZE_T n = ::VirtualQueryEx(to_handle(handle_),
                                reinterpret_cast<LPCVOID>(addr), &mbi, sizeof(mbi));
    if (n == 0) {
        LOG_WARN("VirtualQueryEx(0x%llx) failed: GLE=%lu",
                 (unsigned long long)addr, ::GetLastError());
        return Status::HostQueryFailed;
    }
    if (mbi.State == MEM_FREE) return Status::UnmappedRead;
    out.base    = reinterpret_cast<GuestAddr>(mbi.BaseAddress);
    out.size    = static_cast<u64>(mbi.RegionSize);
    out.protect = static_cast<u32>(mbi.Protect);
    out.state   = static_cast<u32>(mbi.State);
    out.type    = static_cast<u32>(mbi.Type);
    return Status::Ok;
}

Status Process::main_module_base(GuestAddr& out) const noexcept {
    if (!is_open()) return Status::HostOpenFailed;
    HMODULE mods[1] = {};
    DWORD needed = 0;
    if (!::EnumProcessModules(to_handle(handle_), mods, sizeof(mods), &needed)) {
        LOG_WARN("EnumProcessModules failed: GLE=%lu", ::GetLastError());
        return Status::HostQueryFailed;
    }
    if (needed < sizeof(HMODULE)) return Status::HostQueryFailed;
    out = reinterpret_cast<GuestAddr>(mods[0]);
    return Status::Ok;
}

// ---- WinProcessMemoryProvider ----------------------------------------------

Status WinProcessMemoryProvider::read(GuestAddr addr, usize size, void* out) noexcept {
    if (proc_ == nullptr || !proc_->is_open()) return Status::HostOpenFailed;
    return proc_->read(addr, size, out);
}

bool WinProcessMemoryProvider::is_mapped(GuestAddr addr) const noexcept {
    if (proc_ == nullptr || !proc_->is_open()) return false;
    ProcessRegion r{};
    return ok(proc_->query(addr, r));
}

} // namespace emu::host
