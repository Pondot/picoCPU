// Cross-check: lazy `stash_*` path must produce identical flag bits to the
// eager `set_flags_*_eager` reference, for every operand size and a sampling
// of input pairs that exercise the edge cases (zero, signs, wraparound,
// boundary values).

#include "test_framework.h"

#include "emu/cpu.h"
#include "emu/types.h"

#include <vector>

using namespace emu;

namespace {

struct FlagBits {
    bool cf, pf, af, zf, sf, of;
};

FlagBits read_arith_flags(const Cpu& c) {
    return FlagBits{c.cf(), c.pf(), c.af(), c.zf(), c.sf(), c.of()};
}

bool flags_equal(const FlagBits& a, const FlagBits& b) {
    return a.cf==b.cf && a.pf==b.pf && a.af==b.af
        && a.zf==b.zf && a.sf==b.sf && a.of==b.of;
}

void print_diff(int& fails, const char* op, u8 size, u64 a, u64 b,
                const FlagBits& lazy, const FlagBits& eager) {
    std::printf("  %s size=%u a=0x%llx b=0x%llx\n", op, size,
                (unsigned long long)a, (unsigned long long)b);
    std::printf("    lazy : cf=%d pf=%d af=%d zf=%d sf=%d of=%d\n",
                lazy.cf, lazy.pf, lazy.af, lazy.zf, lazy.sf, lazy.of);
    std::printf("    eager: cf=%d pf=%d af=%d zf=%d sf=%d of=%d\n",
                eager.cf, eager.pf, eager.af, eager.zf, eager.sf, eager.of);
    ++fails;
}

std::vector<u64> sample_values() {
    return {
        0ull, 1ull, 2ull, 3ull,
        0x7Full, 0x80ull, 0xFFull, 0x100ull,
        0x7FFFull, 0x8000ull, 0xFFFFull,
        0x7FFFFFFFull, 0x80000000ull, 0xFFFFFFFFull,
        0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull,
        0x5555555555555555ull, 0xAAAAAAAAAAAAAAAAull,
        0xDEADBEEFCAFEBABEull, 0x1234567890ABCDEFull,
    };
}

void crosscheck_binary(int& __fails, const char* name,
                       void (*lazy)  (Cpu&, u8, u64, u64, u64),
                       void (*eager) (Cpu&, u8, u64, u64, u64),
                       u64 (*op)(u64, u64)) {
    const u8 sizes[] = {1, 2, 4, 8};
    for (u8 size : sizes) {
        const u64 mask = mask_for(size);
        for (u64 a_raw : sample_values()) {
            for (u64 b_raw : sample_values()) {
                const u64 a = a_raw & mask;
                const u64 b = b_raw & mask;
                const u64 r = op(a, b) & mask;

                Cpu cL, cE;
                lazy (cL, size, a, b, r);
                eager(cE, size, a, b, r);

                const auto fL = read_arith_flags(cL);
                const auto fE = read_arith_flags(cE);
                if (!flags_equal(fL, fE)) {
                    print_diff(__fails, name, size, a, b, fL, fE);
                }
            }
        }
    }
}

void wrap_add  (Cpu& c, u8 sz, u64 a, u64 b, u64 r) { c.stash_add(sz, b, r);            }
void wrap_addE (Cpu& c, u8 sz, u64 a, u64 b, u64 r) { c.set_flags_add_eager(sz, a, b, r);}
void wrap_sub  (Cpu& c, u8 sz, u64 a, u64 b, u64 r) { (void)a; c.stash_sub(sz, b, r);   }
void wrap_subE (Cpu& c, u8 sz, u64 a, u64 b, u64 r) { c.set_flags_sub_eager(sz, a, b, r);}
void wrap_logic (Cpu& c, u8 sz, u64 a, u64 b, u64 r) { (void)a;(void)b; c.stash_logic(sz, r); }
void wrap_logicE(Cpu& c, u8 sz, u64 a, u64 b, u64 r) { (void)a;(void)b; c.set_flags_logical_eager(sz, r); }

u64 do_add (u64 a, u64 b) { return a + b; }
u64 do_sub (u64 a, u64 b) { return a - b; }
u64 do_and (u64 a, u64 b) { return a & b; }
u64 do_or  (u64 a, u64 b) { return a | b; }
u64 do_xor (u64 a, u64 b) { return a ^ b; }

} // namespace

TEST(crosscheck_add) {
    crosscheck_binary(__fails, "ADD", &wrap_add, &wrap_addE, &do_add);
}

TEST(crosscheck_sub) {
    crosscheck_binary(__fails, "SUB", &wrap_sub, &wrap_subE, &do_sub);
}

TEST(crosscheck_and) {
    crosscheck_binary(__fails, "AND", &wrap_logic, &wrap_logicE, &do_and);
}

TEST(crosscheck_or) {
    crosscheck_binary(__fails, "OR",  &wrap_logic, &wrap_logicE, &do_or);
}

TEST(crosscheck_xor) {
    crosscheck_binary(__fails, "XOR", &wrap_logic, &wrap_logicE, &do_xor);
}

TEST(crosscheck_inc_dec_neg) {
    const u8 sizes[] = {1, 2, 4, 8};
    for (u8 size : sizes) {
        const u64 mask = mask_for(size);
        for (u64 v : sample_values()) {
            const u64 a = v & mask;
            // INC
            {
                Cpu cL, cE;
                // Both: pre-set CF=1 then INC must preserve it.
                cL.set_cf(true); cE.set_cf(true);
                const u64 r = (a + 1) & mask;
                cL.stash_inc(size, r);
                cE.set_flags_inc_eager(size, a, r);
                const auto fL = read_arith_flags(cL);
                const auto fE = read_arith_flags(cE);
                if (!flags_equal(fL, fE)) print_diff(__fails, "INC", size, a, 0, fL, fE);
            }
            // DEC
            {
                Cpu cL, cE;
                cL.set_cf(true); cE.set_cf(true);
                const u64 r = (a - 1) & mask;
                cL.stash_dec(size, r);
                cE.set_flags_dec_eager(size, a, r);
                const auto fL = read_arith_flags(cL);
                const auto fE = read_arith_flags(cE);
                if (!flags_equal(fL, fE)) print_diff(__fails, "DEC", size, a, 0, fL, fE);
            }
            // NEG
            {
                Cpu cL, cE;
                const u64 r = ((~a) + 1ull) & mask;
                cL.stash_neg(size, a, r);
                cE.set_flags_neg_eager(size, a, r);
                const auto fL = read_arith_flags(cL);
                const auto fE = read_arith_flags(cE);
                if (!flags_equal(fL, fE)) print_diff(__fails, "NEG", size, a, 0, fL, fE);
            }
        }
    }
}

TEST(crosscheck_shifts) {
    const u8 sizes[] = {1, 2, 4, 8};
    for (u8 size : sizes) {
        const u8 width = static_cast<u8>(size * 8);
        for (u64 v : sample_values()) {
            const u64 a = v & mask_for(size);
            for (u8 count : {u8{0}, u8{1}, u8{2}, u8{7}, u8{31}, u8{63}}) {
                if (count >= width && size < 8) continue;  // SHL by ≥ width is fine on x86 but our refs skip
                const u8 c_eff = static_cast<u8>(count & (size == 8 ? 0x3F : 0x1F));

                // SHL
                {
                    Cpu cL, cE;
                    const u64 r = (c_eff >= width) ? 0 : ((a << c_eff) & mask_for(size));
                    cL.stash_shl(size, a, c_eff, r);
                    cE.set_flags_shl_eager(size, a, c_eff, r);
                    if (!flags_equal(read_arith_flags(cL), read_arith_flags(cE))) {
                        print_diff(__fails, "SHL", size, a, c_eff,
                                   read_arith_flags(cL), read_arith_flags(cE));
                    }
                }
                // SHR
                {
                    Cpu cL, cE;
                    const u64 r = (c_eff >= width) ? 0 : (a >> c_eff);
                    cL.stash_shr(size, a, c_eff, r);
                    cE.set_flags_shr_eager(size, a, c_eff, r);
                    if (!flags_equal(read_arith_flags(cL), read_arith_flags(cE))) {
                        print_diff(__fails, "SHR", size, a, c_eff,
                                   read_arith_flags(cL), read_arith_flags(cE));
                    }
                }
            }
        }
    }
}
