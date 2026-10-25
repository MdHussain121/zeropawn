"""Smoke test: import zeropawn_cpp and verify it exposes version()."""


def test_import():
    import zeropawn_cpp

    v = zeropawn_cpp.version()
    assert isinstance(v, str)
    assert "phase1" in v or "0." in v, f"Unexpected version: {v}"
    print(f"zeropawn_cpp.version() = {v}")
