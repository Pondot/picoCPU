// Emulator: Status -> string mapping.

#include "emu/error.h"

namespace emu {

std::string_view to_string(Status s) noexcept {
    switch (s) {
        case Status::Ok:                  return "Ok";
        case Status::InvalidArgument:     return "InvalidArgument";
        case Status::OutOfMemory:         return "OutOfMemory";
        case Status::NotImplemented:      return "NotImplemented";
        case Status::Internal:            return "Internal";
        case Status::UnmappedRead:        return "UnmappedRead";
        case Status::UnmappedWrite:       return "UnmappedWrite";
        case Status::ProtectionViolation: return "ProtectionViolation";
        case Status::ProviderFailure:     return "ProviderFailure";
        case Status::InvalidInstruction:  return "InvalidInstruction";
        case Status::TruncatedInstruction:return "TruncatedInstruction";
        case Status::UnsupportedPrefix:   return "UnsupportedPrefix";
        case Status::UnsupportedOpcode:   return "UnsupportedOpcode";
        case Status::HostOpenFailed:      return "HostOpenFailed";
        case Status::HostQueryFailed:     return "HostQueryFailed";
        case Status::HostReadFailed:      return "HostReadFailed";
        case Status::ProcessNotFound:     return "ProcessNotFound";
        case Status::FunctionNotFound:    return "FunctionNotFound";
    }
    return "Unknown";
}

} // namespace emu
