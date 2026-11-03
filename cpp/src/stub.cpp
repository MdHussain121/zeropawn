// Native version provider for the zeropawn_cpp extension (bound as
// zeropawn_cpp.version()). The version is the single package version declared
// in pyproject.toml (S6-08: CLI and native core must report the same one).
// Historical note (S1-07): this translation unit previously lived behind a
// "placeholder" stub.h comment and returned "0.1.0-phase1"; the misleading
// labels are retired. The file keeps its historical name and the stub.h
// header to avoid churning the build graph; the header's own "placeholder"
// comment is updated to match.
#include "zeropawn/stub.h"

namespace zeropawn {
    const char* version() { return "0.1.0"; }
}
