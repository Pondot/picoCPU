// Emulator: logging.
//
// Ring-buffered, lock-free producer side, single flush thread on the consumer
// side. TRACE/DEBUG compile to nothing in Release.

#pragma once

#include "emu/types.h"

#include <cstdarg>
#include <cstdio>

namespace emu::log {

enum class Level : u8 {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Off   = 5,
};

// Initialize the logger. Safe to call once at program start.
void init(Level threshold = Level::Info);

// Shut down: flush, join flush thread, free resources.
void shutdown();

// Adjust runtime threshold.
void set_level(Level lvl) noexcept;
Level get_level() noexcept;

// Internal: emit a formatted line. Use the macros below, not this directly.
void emit(Level lvl, const char* file, int line, const char* fmt, ...) noexcept;

// Force-flush the ring (debugging / tests).
void flush();

} // namespace emu::log

// ---- Macros -----------------------------------------------------------------

#define EMU_LOG_IMPL(LVL, FMT, ...) \
    ::emu::log::emit((LVL), __FILE__, __LINE__, FMT, ##__VA_ARGS__)

#if defined(NDEBUG)
    #define LOG_TRACE(FMT, ...) ((void)0)
    #define LOG_DEBUG(FMT, ...) ((void)0)
#else
    #define LOG_TRACE(FMT, ...) EMU_LOG_IMPL(::emu::log::Level::Trace, FMT, ##__VA_ARGS__)
    #define LOG_DEBUG(FMT, ...) EMU_LOG_IMPL(::emu::log::Level::Debug, FMT, ##__VA_ARGS__)
#endif

#define LOG_INFO(FMT, ...)  EMU_LOG_IMPL(::emu::log::Level::Info,  FMT, ##__VA_ARGS__)
#define LOG_WARN(FMT, ...)  EMU_LOG_IMPL(::emu::log::Level::Warn,  FMT, ##__VA_ARGS__)
#define LOG_ERROR(FMT, ...) EMU_LOG_IMPL(::emu::log::Level::Error, FMT, ##__VA_ARGS__)
