// Emulator: status / error codes.
//
// Hot paths return `Status` (an enum-class). Faults during emulation use the
// dedicated `Fault` type defined in fault.h -- not error.h -- because the
// dispatcher needs richer context (faulting address, kind) than a single enum.

#pragma once

#include <string_view>

namespace emu {

enum class Status : int {
    Ok = 0,

    // Generic
    InvalidArgument,
    OutOfMemory,
    NotImplemented,
    Internal,

    // Memory provider
    UnmappedRead,
    UnmappedWrite,
    ProtectionViolation,
    ProviderFailure,        // backend (e.g. RPM) returned an error

    // Decoder
    InvalidInstruction,
    TruncatedInstruction,
    UnsupportedPrefix,
    UnsupportedOpcode,

    // Host (Windows)
    HostOpenFailed,
    HostQueryFailed,
    HostReadFailed,

    // Tester / process
    ProcessNotFound,
    FunctionNotFound,
};

constexpr bool ok(Status s) { return s == Status::Ok; }
constexpr bool fail(Status s) { return s != Status::Ok; }

std::string_view to_string(Status s) noexcept;

} // namespace emu
