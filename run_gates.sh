#!/usr/bin/env bash
# Spawn target.exe, read its address dump, run every live gate, kill target.
set -e
trap 'kill $TPID 2>/dev/null; exit' EXIT INT TERM

OUT=./target_out.txt
rm -f "$OUT"
./build/Release/bin/target.exe > "$OUT" 2>&1 &
TPID=$!

# Wait until target has printed all gate lines (look for selfmod_fn).
for i in {1..30}; do
    if grep -q "^selfmod_fn:" "$OUT" 2>/dev/null; then break; fi
    sleep 0.2
done

if ! grep -q "^selfmod_fn:" "$OUT"; then
    echo "target failed to start"
    cat "$OUT"
    exit 1
fi

PID=$(awk -F'pid=' '/^target:/ {split($2,a," "); print a[1]}' "$OUT")
echo "===== target PID=$PID"
cat "$OUT"
echo "----------"

extract_field () {  # usage: extract_field <line-prefix> <field-prefix>
    awk -v lp="$1" -v fp="$2" '
        $0 ~ "^" lp {
            for (i = 1; i <= NF; ++i) {
                if (index($i, fp) == 1) {
                    val = substr($i, length(fp) + 1)
                    print val
                    exit
                }
            }
        }
    ' "$OUT"
}

# --- Gate 1: simple mixer
echo; echo "=== gate1: simple mixer ==="
ADDR=$(extract_field "target:" "fn=")
SZ=$(extract_field "target:" "size=")
EXP=$(extract_field "target:" "expected(seed=1234)=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --emulate --seed 1234 --expected "$EXP" 2>&1 | tail -8

# --- Gate 2: GS:[0x30]
echo; echo "=== gate2: GS:[0x30] ==="
ADDR=$(extract_field "teb_fn:" "addr=")
SZ=$(extract_field "teb_fn:" "size=")
TEB=$(extract_field "teb_fn:" "teb=")
TID=$(extract_field "teb_fn:" "tid=")
./build/Release/bin/tester.exe --pid "$PID" --tid "$TID" --addr "$ADDR" --bytes "$SZ" --gs-test --expected "$TEB" 2>&1 | tail -8

# --- Gate 3: stack frame
echo; echo "=== gate3: stack frame ==="
ADDR=$(extract_field "stack_fn:" "addr=")
SZ=$(extract_field "stack_fn:" "size=")
EXP=$(extract_field "stack_fn:" "expected(seed=1234)=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --stack-test --seed 1234 --expected "$EXP" 2>&1 | tail -8

# --- Gate 4: polymorphic
echo; echo "=== gate4: polymorphic ==="
ADDR=$(extract_field "poly_fn:" "addr=")
SZ=$(extract_field "poly_fn:" "size=")
EXP=$(extract_field "poly_fn:" "expected(seed=1234)=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --stack-test --seed 1234 --expected "$EXP" 2>&1 | tail -8

# --- Gate 5: obfuscated
echo; echo "=== gate5: obfuscated ==="
ADDR=$(extract_field "obf_fn:" "addr=")
SZ=$(extract_field "obf_fn:" "size=")
EXP=$(extract_field "obf_fn:" "expected(seed=1234)=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --stack-test --seed 1234 --expected "$EXP" 2>&1 | tail -8

# --- Gate 6: AVX
echo; echo "=== gate6: AVX ==="
ADDR=$(extract_field "avx_fn:" "addr=")
SZ=$(extract_field "avx_fn:" "size=")
EXP=$(extract_field "avx_fn:" "expected(seed=1234)=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --stack-test --seed 1234 --expected "$EXP" 2>&1 | tail -8

# --- Gate 7: BMI
echo; echo "=== gate7: BMI ==="
ADDR=$(extract_field "bmi_fn:" "addr=")
SZ=$(extract_field "bmi_fn:" "size=")
EXP=$(extract_field "bmi_fn:" "expected(seed=1234)=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --stack-test --seed 1234 --expected "$EXP" 2>&1 | tail -8

# --- Gate 8: entity walker (NEW)
echo; echo "=== gate8: entity walker ==="
ADDR=$(extract_field "entity_fn:" "addr=")
SZ=$(extract_field "entity_fn:" "size=")
ARR=$(extract_field "entity_fn:" "arr=")
CNT=$(extract_field "entity_fn:" "count=")
EXP=$(extract_field "entity_fn:" "expected=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --entity-test --arr "$ARR" --count "$CNT" --expected "$EXP" 2>&1 | tail -8

# --- Gate 9: self-modifying code (NEW)
echo; echo "=== gate9: self-modifying code ==="
ADDR=$(extract_field "selfmod_fn:" "addr=")
SZ=$(extract_field "selfmod_fn:" "size=")
THUNK=$(extract_field "selfmod_fn:" "thunk=")
EXP=$(extract_field "selfmod_fn:" "expected(seed=1234)=")
./build/Release/bin/tester.exe --pid "$PID" --addr "$ADDR" --bytes "$SZ" --selfmod-test --thunk "$THUNK" --thunk-size 4096 --seed 1234 --expected "$EXP" 2>&1 | tail -8
