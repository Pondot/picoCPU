// Win64 syscall canned dispatch.
//
// Installs an INSN hook (filter_op = OpKind::Syscall) that intercepts SYSCALL
// instructions, reads the syscall number from RAX, and dispatches a small
// canned table that mimics common ntdll behavior just enough to keep
// user-mode functions running:
//
//   NtClose                     -> success
//   NtAllocateVirtualMemory     -> fills out *BaseAddress, *RegionSize, returns 0
//   NtFreeVirtualMemory         -> success
//   NtProtectVirtualMemory      -> success
//   NtQueryInformationProcess   -> minimal info, success
//   NtTerminateThread           -> halts the Cpu cleanly
//
// Numbers vary across Windows versions; users can override individual entries
// or set numbers per OS build via `set_syscall_number()`.
//
// Real workloads that need full ntdll semantics should install their own
// INSN hook with richer behavior -- this is a "make most user code keep
// running" stub.

#pragma once

#include "emu/emu.h"
#include "emu/types.h"

namespace emu {

// Install the canned-dispatch hook on `e`. Returns the hook id so the
// caller can remove it later. Multiple calls reuse the same handler.
HookManager::HookId install_default_syscalls(Emulator& e) noexcept;

// Override the syscall number for a named entry. Useful when the target's
// Windows build uses a different number than our defaults.
enum class SyscallEntry : u32 {
    NtClose,
    NtAllocateVirtualMemory,
    NtFreeVirtualMemory,
    NtProtectVirtualMemory,
    NtQueryInformationProcess,
    NtTerminateThread,
    _Count,
};

void set_syscall_number(SyscallEntry e, u32 number) noexcept;

} // namespace emu
