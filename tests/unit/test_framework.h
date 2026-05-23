// Tiny ad-hoc unit-test framework. No external deps. Run with: tests_unit.exe
//
// Macros:
//   TEST(name) { ... }         declares a test, auto-registered
//   EXPECT(cond)               soft check -- keeps running
//   EXPECT_EQ(a, b)            soft check with printed diff
//   EXPECT_HEX(a, b)           same but prints as hex
//   REQUIRE(cond)              hard check -- aborts current test
//
// Failures print file:line and the offending values. Process exits non-zero
// if any test failed.

#pragma once

#include "emu/types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tu {

struct Failure { const char* file; int line; const char* msg; };

struct Test {
    const char* name;
    void      (*fn)(int&);   // fn(failures) -- increments on fail
};

inline std::vector<Test>& registry() { static std::vector<Test> v; return v; }

inline int run_all() {
    int total = 0, fails = 0;
    for (const auto& t : registry()) {
        ++total;
        int f = 0;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn(f);
        if (f) {
            std::printf("[ FAIL ] %s (%d failure(s))\n", t.name, f);
            ++fails;
        } else {
            std::printf("[  OK  ] %s\n", t.name);
        }
    }
    std::printf("---- %d tests, %d failed ----\n", total, fails);
    return fails;
}

struct AutoRegister {
    AutoRegister(const char* name, void (*fn)(int&)) {
        registry().push_back({name, fn});
    }
};

} // namespace tu

#define TU_CONCAT_(a, b) a##b
#define TU_CONCAT(a, b) TU_CONCAT_(a, b)

#define TEST(NAME)                                                              \
    static void TU_CONCAT(test_, NAME)(int& __fails);                           \
    static ::tu::AutoRegister TU_CONCAT(reg_, NAME)(#NAME, &TU_CONCAT(test_, NAME)); \
    static void TU_CONCAT(test_, NAME)(int& __fails)

#define EXPECT(COND)                                                            \
    do {                                                                        \
        if (!(COND)) {                                                          \
            std::printf("  EXPECT(%s) FAILED at %s:%d\n", #COND, __FILE__, __LINE__); \
            ++__fails;                                                          \
        }                                                                       \
    } while (0)

#define EXPECT_EQ(A, B)                                                         \
    do {                                                                        \
        auto __a = (A); auto __b = (B);                                         \
        if (!(__a == __b)) {                                                    \
            std::printf("  EXPECT_EQ(%s, %s) FAILED at %s:%d  a=%lld b=%lld\n", \
                        #A, #B, __FILE__, __LINE__,                             \
                        (long long)__a, (long long)__b);                        \
            ++__fails;                                                          \
        }                                                                       \
    } while (0)

#define EXPECT_HEX(A, B)                                                        \
    do {                                                                        \
        auto __a = (A); auto __b = (B);                                         \
        if (!(__a == __b)) {                                                    \
            std::printf("  EXPECT_HEX(%s, %s) FAILED at %s:%d  a=0x%llx b=0x%llx\n", \
                        #A, #B, __FILE__, __LINE__,                             \
                        (unsigned long long)__a, (unsigned long long)__b);      \
            ++__fails;                                                          \
        }                                                                       \
    } while (0)

#define REQUIRE(COND)                                                           \
    do {                                                                        \
        if (!(COND)) {                                                          \
            std::printf("  REQUIRE(%s) FAILED at %s:%d (abort)\n", #COND, __FILE__, __LINE__); \
            ++__fails;                                                          \
            return;                                                             \
        }                                                                       \
    } while (0)
