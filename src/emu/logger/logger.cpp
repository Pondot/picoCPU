// Emulator: logger implementation.
//
// Ring buffer of fixed-size records. Producers fill records atomically and
// stamp them as ready; a flush thread drains in order and writes to stderr.
// On overflow we drop oldest unread records (writer wins).

#include "emu/logger.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace emu::log {

namespace {

constexpr usize RING_CAPACITY = 1u << 14;        // 16384 records (power of two)
constexpr usize RECORD_MSG    = 240;             // bytes of formatted message
constexpr usize MAX_FILE_LEN  = 64;

struct Record {
    std::atomic<u64> seq;                        // even = empty; odd = published
    Level            level;
    u32              tid;
    i64              ts_ns;
    int              line;
    char             file[MAX_FILE_LEN];
    char             msg[RECORD_MSG];
};

struct Ring {
    Record                   slots[RING_CAPACITY];
    std::atomic<u64>         write_seq{0};
    std::atomic<u64>         read_seq{0};
    std::atomic<Level>       threshold{Level::Info};
    std::atomic<bool>        running{false};
    std::thread              flush_thr;
    std::mutex               wake_mtx;
    std::condition_variable  wake_cv;
};

Ring& ring() {
    static Ring r{};
    return r;
}

const char* level_str(Level l) noexcept {
    switch (l) {
        case Level::Trace: return "TRACE";
        case Level::Debug: return "DEBUG";
        case Level::Info:  return "INFO ";
        case Level::Warn:  return "WARN ";
        case Level::Error: return "ERROR";
        default:           return "?????";
    }
}

i64 now_ns() noexcept {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

u32 current_tid() noexcept {
    // Cheap, monotonic per-thread id derived from std::thread::id hash.
    static thread_local const u32 tid = static_cast<u32>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFFFFFu);
    return tid;
}

void drain_one(const Record& rec) noexcept {
    std::fprintf(stderr, "[%lld] [%5u] [%s] %s:%d: %s\n",
                 static_cast<long long>(rec.ts_ns),
                 rec.tid,
                 level_str(rec.level),
                 rec.file,
                 rec.line,
                 rec.msg);
}

void flush_loop() {
    auto& r = ring();
    while (r.running.load(std::memory_order_acquire)) {
        u64 head = r.read_seq.load(std::memory_order_relaxed);
        u64 tail = r.write_seq.load(std::memory_order_acquire);

        bool did_work = false;
        while (head < tail) {
            Record& slot = r.slots[head % RING_CAPACITY];
            u64 want = (head + 1) * 2;            // expected published seq
            u64 cur  = slot.seq.load(std::memory_order_acquire);
            if (cur != want) break;               // not yet published
            drain_one(slot);
            // Mark empty so the slot can be reused.
            slot.seq.store(want + 1, std::memory_order_release); // even == empty next cycle
            ++head;
            did_work = true;
        }
        r.read_seq.store(head, std::memory_order_release);

        if (!did_work) {
            std::unique_lock<std::mutex> lk(r.wake_mtx);
            r.wake_cv.wait_for(lk, std::chrono::milliseconds(50));
        }
    }
    // Final drain on shutdown.
    u64 head = r.read_seq.load(std::memory_order_relaxed);
    u64 tail = r.write_seq.load(std::memory_order_acquire);
    while (head < tail) {
        Record& slot = r.slots[head % RING_CAPACITY];
        u64 want = (head + 1) * 2;
        u64 cur  = slot.seq.load(std::memory_order_acquire);
        if (cur != want) { ++head; continue; }
        drain_one(slot);
        ++head;
    }
    std::fflush(stderr);
}

} // namespace

void init(Level threshold) {
    auto& r = ring();
    if (r.running.exchange(true, std::memory_order_acq_rel)) return; // already up
    r.threshold.store(threshold, std::memory_order_release);
    r.write_seq.store(0, std::memory_order_release);
    r.read_seq.store(0, std::memory_order_release);
    for (auto& s : r.slots) {
        s.seq.store(0, std::memory_order_relaxed);
    }
    r.flush_thr = std::thread(flush_loop);
}

void shutdown() {
    auto& r = ring();
    if (!r.running.exchange(false, std::memory_order_acq_rel)) return;
    {
        std::lock_guard<std::mutex> lk(r.wake_mtx);
        r.wake_cv.notify_all();
    }
    if (r.flush_thr.joinable()) r.flush_thr.join();
}

void set_level(Level lvl) noexcept {
    ring().threshold.store(lvl, std::memory_order_release);
}

Level get_level() noexcept {
    return ring().threshold.load(std::memory_order_acquire);
}

void flush() {
    auto& r = ring();
    // Wake the flusher and yield briefly so it can drain.
    {
        std::lock_guard<std::mutex> lk(r.wake_mtx);
        r.wake_cv.notify_all();
    }
    for (int i = 0; i < 100; ++i) {
        if (r.read_seq.load(std::memory_order_acquire) ==
            r.write_seq.load(std::memory_order_acquire)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::fflush(stderr);
}

void emit(Level lvl, const char* file, int line, const char* fmt, ...) noexcept {
    auto& r = ring();
    if (lvl < r.threshold.load(std::memory_order_relaxed)) return;
    if (!r.running.load(std::memory_order_acquire)) {
        // Fast path before init / after shutdown: write straight to stderr.
        va_list ap;
        va_start(ap, fmt);
        std::fprintf(stderr, "[%s] %s:%d: ", level_str(lvl), file, line);
        std::vfprintf(stderr, fmt, ap);
        std::fputc('\n', stderr);
        va_end(ap);
        return;
    }

    u64 my_seq = r.write_seq.fetch_add(1, std::memory_order_acq_rel);
    Record& slot = r.slots[my_seq % RING_CAPACITY];

    // Wait for the slot to be free if we've lapped the reader. We tolerate
    // ~1ms of contention; beyond that we drop the message rather than block
    // the hot path.
    for (int spins = 0; spins < 16; ++spins) {
        u64 cur = slot.seq.load(std::memory_order_acquire);
        // Slot is free if cur is even AND below the seq we're about to publish.
        if ((cur & 1ull) == 0 && cur <= my_seq * 2) break;
        if (spins > 4) std::this_thread::yield();
    }

    slot.level = lvl;
    slot.tid   = current_tid();
    slot.ts_ns = now_ns();
    slot.line  = line;

    // Truncate filename to just the last path segment for readability.
    const char* base = file;
    for (const char* p = file; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    std::strncpy(slot.file, base, MAX_FILE_LEN - 1);
    slot.file[MAX_FILE_LEN - 1] = '\0';

    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(slot.msg, RECORD_MSG, fmt, ap);
    va_end(ap);
    slot.msg[RECORD_MSG - 1] = '\0';

    // Publish: odd seq = ready.
    slot.seq.store((my_seq + 1) * 2, std::memory_order_release);

    // Nudge the flusher if it's sleeping.
    if (lvl >= Level::Warn) {
        std::lock_guard<std::mutex> lk(r.wake_mtx);
        r.wake_cv.notify_one();
    }
}

} // namespace emu::log
