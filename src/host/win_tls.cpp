// TEB lookup via NtQueryInformationThread + Toolhelp32 thread enumeration.

#include "win_tls.h"

#include "emu/logger.h"

#include <windows.h>
#include <tlhelp32.h>

namespace emu::host {

namespace {

// Layouts borrowed from <winternl.h> -- we redefine the minimum we need so we
// don't drag the rest of the private headers in.
struct CLIENT_ID_MINI {
    HANDLE UniqueProcess;
    HANDLE UniqueThread;
};

struct THREAD_BASIC_INFO {
    LONG          ExitStatus;
    PVOID         TebBaseAddress;
    CLIENT_ID_MINI ClientId;
    KAFFINITY     AffinityMask;
    ULONG         Priority;
    LONG          BasePriority;
};

constexpr int ThreadBasicInformation = 0;

using NtQueryInformationThread_t = LONG (NTAPI*)(
    HANDLE ThreadHandle, int ThreadInformationClass,
    PVOID  ThreadInformation, ULONG ThreadInformationLength,
    PULONG ReturnLength);

NtQueryInformationThread_t resolve_nqit() noexcept {
    HMODULE h = ::GetModuleHandleW(L"ntdll.dll");
    if (!h) return nullptr;
    return reinterpret_cast<NtQueryInformationThread_t>(
        ::GetProcAddress(h, "NtQueryInformationThread"));
}

// Pick the lowest-TID thread belonging to `pid`. Returns 0 if not found.
u32 main_thread_of(u32 pid) noexcept {
    HANDLE snap = ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    u32 best_tid = 0;
    if (::Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid) {
                if (best_tid == 0 || te.th32ThreadID < best_tid) {
                    best_tid = te.th32ThreadID;
                }
            }
        } while (::Thread32Next(snap, &te));
    }
    ::CloseHandle(snap);
    return best_tid;
}

} // namespace

Status query_thread_teb(u32 pid, u32 tid, GuestAddr& out_teb) noexcept {
    out_teb = 0;
    if (tid == 0) {
        tid = main_thread_of(pid);
        if (tid == 0) {
            LOG_WARN("query_thread_teb: no threads found for pid %u", pid);
            return Status::HostQueryFailed;
        }
    }

    HANDLE h = ::OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid);
    if (h == nullptr) {
        h = ::OpenThread(THREAD_QUERY_INFORMATION, FALSE, tid);
    }
    if (h == nullptr) {
        LOG_WARN("OpenThread(tid=%u) failed: GLE=%lu", tid, ::GetLastError());
        return Status::HostQueryFailed;
    }

    auto nqit = resolve_nqit();
    if (nqit == nullptr) {
        ::CloseHandle(h);
        LOG_WARN("NtQueryInformationThread not resolvable");
        return Status::HostQueryFailed;
    }

    THREAD_BASIC_INFO tbi{};
    ULONG ret = 0;
    LONG  rc  = nqit(h, ThreadBasicInformation, &tbi, sizeof(tbi), &ret);
    ::CloseHandle(h);

    if (rc < 0) {  // NTSTATUS error
        LOG_WARN("NtQueryInformationThread(tid=%u) NTSTATUS=0x%08lx", tid, (unsigned long)rc);
        return Status::HostQueryFailed;
    }

    out_teb = reinterpret_cast<GuestAddr>(tbi.TebBaseAddress);
    if (out_teb == 0) {
        return Status::HostQueryFailed;
    }
    return Status::Ok;
}

} // namespace emu::host
