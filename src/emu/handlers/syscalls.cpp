// Syscall canned dispatch implementation.

#include "emu/syscalls.h"

#include "emu/cpu.h"
#include "emu/hooks.h"
#include "emu/logger.h"

#include <array>

namespace emu {

namespace {

// Default numbers -- Windows 10 22H2 / 11 (close enough that most test
// targets land in the right ballpark). User code calls set_syscall_number()
// to retarget for older OS builds.
u32 g_syscall_numbers[static_cast<u8>(SyscallEntry::_Count)] = {
    /* NtClose                   */ 0x000F,
    /* NtAllocateVirtualMemory   */ 0x0018,
    /* NtFreeVirtualMemory       */ 0x001E,
    /* NtProtectVirtualMemory    */ 0x0050,
    /* NtQueryInformationProcess */ 0x0019,
    /* NtTerminateThread         */ 0x0024,
};

constexpr u64 STATUS_SUCCESS = 0;

void canned_syscall_dispatch(void* /*user*/, Cpu& cpu, const Insn& insn) {
    if (insn.kind != OpKind::Syscall) return;

    const u32 sysno = cpu.r32(reg::RAX);
    auto matches = [&](SyscallEntry e) {
        return sysno == g_syscall_numbers[static_cast<u8>(e)];
    };

    if (matches(SyscallEntry::NtClose)) {
        cpu.set_r64(reg::RAX, STATUS_SUCCESS);
        cpu.halt();    // SYSCALL halts emulation; caller's Cpu::unhalt is needed to continue
        return;
    }
    if (matches(SyscallEntry::NtAllocateVirtualMemory)) {
        // Win64 ABI: args in RCX, RDX, R8, R9 + stack. For syscall the kernel
        // calling convention shifts to (R10 = first arg). We're emulating
        // user-mode SYSCALL -- the user-mode wrapper moves RCX->R10. We don't
        // attempt to actually allocate; just return success and leave the
        // out-params undisturbed.
        cpu.set_r64(reg::RAX, STATUS_SUCCESS);
        cpu.halt();
        return;
    }
    if (matches(SyscallEntry::NtFreeVirtualMemory)
        || matches(SyscallEntry::NtProtectVirtualMemory)) {
        cpu.set_r64(reg::RAX, STATUS_SUCCESS);
        cpu.halt();
        return;
    }
    if (matches(SyscallEntry::NtQueryInformationProcess)) {
        cpu.set_r64(reg::RAX, STATUS_SUCCESS);
        cpu.halt();
        return;
    }
    if (matches(SyscallEntry::NtTerminateThread)) {
        LOG_INFO("emu: NtTerminateThread -- halting cleanly");
        cpu.set_r64(reg::RAX, STATUS_SUCCESS);
        cpu.halt();
        return;
    }

    // Unknown syscall -- let the dispatcher take the default fault path.
    LOG_WARN("emu: unhandled syscall %u -- set_syscall_number or install a custom INSN hook", sysno);
}

} // namespace

HookManager::HookId install_default_syscalls(Emulator& e) noexcept {
    Hook h{};
    h.kind      = HookKind::Insn;
    h.callback  = reinterpret_cast<void*>(&canned_syscall_dispatch);
    h.user_data = nullptr;
    h.begin     = 0;
    h.end       = 0;
    h.filter_op = OpKind::Syscall;
    return e.hooks().add(h);
}

void set_syscall_number(SyscallEntry which, u32 number) noexcept {
    g_syscall_numbers[static_cast<u8>(which)] = number;
}

} // namespace emu
