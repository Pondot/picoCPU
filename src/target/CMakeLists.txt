add_executable(target target.cpp)

# Don't let the linker reorder/fold target_fn / target_fn_end -- we depend on
# their addresses to bracket the function bytes.
target_link_options(target PRIVATE /OPT:NOICF /OPT:NOREF)

# Disable LTCG: we want target_fn's instructions to come straight from /O2
# codegen without cross-module inlining surprises.
target_compile_options(target PRIVATE /GL-)
target_link_options(target PRIVATE /LTCG:OFF)
