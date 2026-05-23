// Test runner entry point.

#include "test_framework.h"

#include "emu/logger.h"

int main() {
    ::emu::log::init(::emu::log::Level::Warn);
    const int rc = ::tu::run_all();
    ::emu::log::shutdown();
    return rc;
}
