// Host: Windows process attach + memory read.
//
// Lives outside src/emu/ on purpose. The emulator core stays host-agnostic;
// only the WinProcessMemoryProvider here knows about HANDLE / RPM.

#pragma once

#include "emu/error.h"
#include "emu/memory.h"
#include "emu/types.h"

#include <string>

namespace emu::host {

// Region descriptor produced by VirtualQueryEx.
struct ProcessRegion {
    GuestAddr base;
    u64       size;
    u32       protect;   // PAGE_* flags from Windows
    u32       state;     // MEM_COMMIT / MEM_RESERVE / MEM_FREE
    u32       type;      // MEM_IMAGE / MEM_MAPPED / MEM_PRIVATE
};

// Opaque handle to an opened process. RAII; closes on dtor.
class Process {
public:
    Process() = default;
    ~Process();

    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;
    Process(Process&&) noexcept;
    Process& operator=(Process&&) noexcept;

    // Open the target by PID, requesting PROCESS_VM_READ + PROCESS_QUERY_*.
    [[nodiscard]] Status open(u32 pid) noexcept;

    // Read bytes from the target process.
    [[nodiscard]] Status read(GuestAddr addr, usize size, void* out) const noexcept;

    // Query the region that contains `addr`. UnmappedRead if free.
    [[nodiscard]] Status query(GuestAddr addr, ProcessRegion& out) const noexcept;

    [[nodiscard]] bool is_open() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] u32  pid()     const noexcept { return pid_; }

    // Returns the base address of the main module (the .exe image) of the
    // opened process. Status::HostQueryFailed if not yet open.
    [[nodiscard]] Status main_module_base(GuestAddr& out) const noexcept;

private:
    void close_() noexcept;

    void* handle_ = nullptr;   // HANDLE; void* to keep Win headers out of the public header
    u32   pid_    = 0;
};

// MemoryProvider backed by a `Process`. Borrows the Process -- does not own it.
class WinProcessMemoryProvider final : public MemoryProvider {
public:
    explicit WinProcessMemoryProvider(const Process& proc) noexcept : proc_(&proc) {}

    [[nodiscard]] Status read(GuestAddr addr, usize size, void* out) noexcept override;
    [[nodiscard]] bool is_mapped(GuestAddr addr) const noexcept override;

private:
    const Process* proc_;
};

} // namespace emu::host
