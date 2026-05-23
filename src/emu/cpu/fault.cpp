// Fault name + Status->FaultKind helpers.

#include "emu/fault.h"

namespace emu {

const char* fault_kind_name(FaultKind k) noexcept {
    switch (k) {
        case FaultKind::None:               return "None";
        case FaultKind::PageFault:          return "#PF";
        case FaultKind::GeneralProtection:  return "#GP";
        case FaultKind::InvalidOpcode:      return "#UD";
        case FaultKind::DivideError:        return "#DE";
        case FaultKind::Breakpoint:         return "#BP";
        case FaultKind::Overflow:           return "#OF";
        case FaultKind::BoundRange:         return "#BR";
        case FaultKind::DeviceNotAvailable: return "#NM";
        case FaultKind::DoubleFault:        return "#DF";
        case FaultKind::InvalidTss:         return "#TS";
        case FaultKind::SegmentNotPresent:  return "#NP";
        case FaultKind::StackFault:         return "#SS";
        case FaultKind::AlignmentCheck:     return "#AC";
        case FaultKind::ProviderFailure:    return "ProviderFailure";
        case FaultKind::NotImplemented:     return "NotImplemented";
        case FaultKind::InternalError:      return "InternalError";
    }
    return "?";
}

FaultKind kind_from_status(Status s) noexcept {
    switch (s) {
        case Status::UnmappedRead:
        case Status::UnmappedWrite:
        case Status::HostReadFailed:
            return FaultKind::PageFault;
        case Status::ProtectionViolation:
            return FaultKind::GeneralProtection;
        case Status::InvalidInstruction:
        case Status::UnsupportedOpcode:
        case Status::UnsupportedPrefix:
            return FaultKind::InvalidOpcode;
        case Status::ProviderFailure:
            return FaultKind::ProviderFailure;
        case Status::NotImplemented:
            return FaultKind::NotImplemented;
        case Status::Internal:
            return FaultKind::InternalError;
        case Status::TruncatedInstruction:
            return FaultKind::InvalidOpcode;
        default:
            return FaultKind::InternalError;
    }
}

} // namespace emu
