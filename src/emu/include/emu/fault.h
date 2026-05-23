// Emulator fault model.
//
// `FaultKind` mirrors the x86 hardware fault vectors we actually care about.
// The dispatcher latches a `Fault` on the Cpu when a handler or memory
// provider fails; emulation halts; the optional fault hook is invoked.

#pragma once

#include "emu/error.h"
#include "emu/types.h"

namespace emu {

enum class FaultKind : u16 {
    None = 0,

    PageFault,           // #PF -- unmapped or wrong-perm memory access
    GeneralProtection,   // #GP -- segment violation, privileged op in user mode
    InvalidOpcode,       // #UD -- unrecognized opcode / unsupported prefix combo
    DivideError,         // #DE -- div/idiv by zero or overflow
    Breakpoint,          // #BP -- INT3
    Overflow,            // #OF -- INTO triggered
    BoundRange,          // #BR -- BOUND check failed
    DeviceNotAvailable,  // #NM -- FP op without CR0.EM/TS handling
    DoubleFault,         // #DF
    InvalidTss,          // #TS
    SegmentNotPresent,   // #NP
    StackFault,          // #SS
    AlignmentCheck,      // #AC

    // Non-architectural -- emulator-internal conditions.
    ProviderFailure,     // host memory backend (RPM) returned an error
    NotImplemented,      // hit an opcode/feature we haven't built yet
    InternalError,       // invariant violated inside the emulator
};

const char* fault_kind_name(FaultKind k) noexcept;

struct Fault {
    FaultKind   kind   = FaultKind::None;
    GuestAddr   rip    = 0;   // RIP of the instruction that caused the fault
    GuestAddr   addr   = 0;   // for #PF: the offending memory address; else 0
    Status      reason = Status::Ok;   // underlying Status when applicable
    const char* note   = nullptr;      // optional diagnostic string, lifetime-static

    [[nodiscard]] bool is_set() const noexcept { return kind != FaultKind::None; }
};

// Heuristic Status -> FaultKind for migrating older set_fault call sites.
FaultKind kind_from_status(Status s) noexcept;

} // namespace emu
