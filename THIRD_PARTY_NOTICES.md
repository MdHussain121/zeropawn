# Third-Party Software Notices and Licenses

ZeroPawn incorporates or depends on third-party open-source software. This document provides notice and attribution for these components.

---

## 1. chess-library (Disservin / Niklas Fiekas)

- **Source:** Vendored in `cpp/vendor/chess-library/`
- **License:** MIT License
- **Copyright:** Copyright (c) 2023 Disservin, Niklas Fiekas

```text
MIT License

Copyright (c) 2023 Disservin, Niklas Fiekas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

*Erratum (audit S1-01, 2026-11-02):* the vendored `chess.hpp` is **not**
byte-identical to upstream — it carries exactly one local functional patch
(`Square::is_valid()` also rejecting negative squares, which the encoder's
en-passant plane guard depends on). The vendored file's "AUTO GENERATED" banner
is therefore not literally true. See
`cpp/vendor/chess-library/LOCAL_PATCHES.md` for the exact hunk, the vendored
base commit (`61e0a694`, 0.9.4), and upgrade guidance.

---

## 2. python-chess (Niklas Fiekas)

- **Nature:** Runtime Python dependency (`python-chess>=1.999`; PyPI distribution `chess`, e.g. 1.11.2)
- **License:** GNU General Public License v3.0 or later (GPL-3.0-or-later; upstream metadata: `GPL-3.0+`)
- **Copyright:** Copyright (c) 2012-2023 Niklas Fiekas
- **Upstream Repository:** https://github.com/niklasf/python-chess

*Notice regarding combined distributions:*  
`python-chess` is distributed under the GNU General Public License v3.0. ZeroPawn's original source code is licensed under the MIT License. If ZeroPawn is bundled or distributed together with `python-chess` (such as in standalone executable packages or pre-built binary distributions), the combined work as a whole is subject to the terms of the GNU General Public License v3.0.

---

## 3. pybind11

- **Nature:** Compile-time C++/Python binding header library (CMake FetchContent, `GIT_TAG v2.13.6`); its compiled object code is embedded in the shipped `zeropawn_cpp` native extension
- **License:** BSD 3-Clause License
- **Copyright:** Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.
- **Upstream Repository:** https://github.com/pybind/pybind11

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 4. Stockfish

- **Nature:** External engine executable invoked out-of-process via standard UCI pipes (`tools/stockfish.exe` or system PATH)
- **License:** GNU General Public License v3.0 (GPLv3)
- **Copyright:** Copyright (c) 2008-2024 Stockfish Developers
- **Upstream Repository:** https://github.com/official-stockfish/Stockfish

*Notice regarding process separation:*  
Stockfish is an external standalone program executed over standard pipes using the Universal Chess Interface (UCI) protocol. It is not statically or dynamically linked into ZeroPawn's binary or address space.

---

## 5. PyTorch & NumPy

- **PyTorch:** BSD-style license. Copyright (c) 2016-present Meta Platforms, Inc. / PyTorch contributors.
- **NumPy:** BSD 3-Clause License. Copyright (c) 2005-2024 NumPy Developers.

---

## 6. libwinpthread-1.dll (MinGW-w64 winpthreads)

- **Nature:** Redistributed binary runtime DLL. The functional wheel
  (`scripts/build_wheel.py`) ships the MinGW-w64 POSIX threads runtime beside
  the first-party `zeropawn_cpp` native extension (`python/_native/`, also
  injected at the site-packages root). The DLL is copied verbatim from the
  MinGW-w64 toolchain used to build the extension and is not modified.
- **License:** MIT (mingw-w64 project) with BSD-3-Clause portions (Lockless Inc.)
- **Copyright:** Copyright (c) 2011 mingw-w64 project; derived portions Copyright (C) 2010 Lockless Inc., All rights reserved. (The DLL's own version resource reads "POSIX WinThreads for Windows", "Copyright (C) MinGW-W64 Project Members 2010-2023".)
- **Upstream Repository:** https://github.com/mingw-w64/mingw-w64 (`mingw-w64-libraries/winpthreads/COPYING`)

```text
Copyright (c) 2011 mingw-w64 project

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

Portions of winpthreads are derived from Lockless Inc.'s "Posix Threads
library for Microsoft Windows", whose redistribution conditions are embedded
in the same upstream `COPYING` file and are reproduced here verbatim
(including the upstream typo truncating `AS IS" AND` to `AS IS" AN`):

```text
(C) 2010 Lockless Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the name of Lockless Inc. nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AN
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 7. Additional Python runtime dependencies

Attribution summaries for the remaining runtime Python packages required by
`pyproject.toml`. Each is used unmodified from PyPI: the wheel declares them
as dependencies and does not redistribute their code.

- **rich:** MIT License. Copyright (c) 2020 Will McGugan. Upstream: https://github.com/Textualize/rich
- **zstandard:** BSD 3-Clause License (License-Expression `BSD-3-Clause`). Copyright (c) 2016, Gregory Szorc. The `zstandard` wheel bundles the native `libzstd` compression library, itself BSD 3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates). Upstream: https://github.com/sergey-dryabzhinsky/python-zstandard and https://github.com/facebook/zstd
- **psutil:** BSD 3-Clause License. Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola. Upstream: https://github.com/giampaolo/psutil
- **tensorboard:** Apache License 2.0 (The TensorFlow Authors / Google LLC). TensorBoard additionally distributes vendored content under Apache 2.0, MIT, and BSD-3 licenses (per its bundled LICENSE). Upstream: https://github.com/tensorflow/tensorboard
