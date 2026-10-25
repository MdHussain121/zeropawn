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

---

## 2. python-chess (Niklas Fiekas)

- **Nature:** Runtime Python dependency (`python-chess>=1.999`)
- **License:** GNU General Public License v3.0 (GPLv3)
- **Copyright:** Copyright (c) 2012-2023 Niklas Fiekas
- **Upstream Repository:** https://github.com/niklasf/python-chess

*Notice regarding combined distributions:*  
`python-chess` is distributed under the GNU General Public License v3.0. ZeroPawn's original source code is licensed under the MIT License. If ZeroPawn is bundled or distributed together with `python-chess` (such as in standalone executable packages or pre-built binary distributions), the combined work as a whole is subject to the terms of the GNU General Public License v3.0.

---

## 3. pybind11

- **Nature:** Compile-time C++/Python binding header library (CMake FetchContent)
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
