#pragma once

// Native version provider declaration (bound as zeropawn_cpp.version()).
// Historical name retained (S1-07/S6-08 remediation 2026-11-03): the file was
// a phase-1 placeholder once; today it fronts the live version export and must
// agree with the package version in pyproject.toml.
namespace zeropawn {
    const char* version();
}
