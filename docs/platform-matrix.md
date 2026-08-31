# Platform matrix

Claims are deliberately separated into design, CI, and physical-machine
verification.

| Platform | Design target | CI verified | Physical machine verified | Current status |
|---|---:|---:|---:|---|
| Windows x64 (VS 2022 baseline) | Yes | Yes | Windows 10 development host | Full OCR, DLL/static package and installed HTTP Demo pass |
| Linux x86_64 (Ubuntu 22.04 baseline) | Yes | Yes | No | Full OCR, package and installed HTTP Demo pass in CI |
| Browser WASM (modern Chromium / Firefox / Safari) | Yes | Yes | No | Standalone HTML runs real OCR repeatedly with stable WASM heap in CI; SIMD128 is enabled by default |
| Linux ARM64 (Ubuntu 22.04 baseline) | Yes | Pending first manual workflow run | No | Native GitHub-hosted ARM64 build, installed-package OCR and ELF/dependency checks are available |
| Linux LoongArch64 (Debian 13 baseline) | Experimental | Pending first manual QEMU run | No | Scalar build and installed-package OCR run under pinned QEMU/container images; customer hardware validation is still required |
| Windows 7 SP1 x64 | Yes | No | No | Planned compatibility validation |
| Windows 7 SP1 x86 | Yes | No | No | Compatibility profile; not a v0.1 blocker |
| Windows 10 x86 build | Design check | No | Windows 10 development host | Full REC path, DLL/static package, and installed Demo pass |

The reusable `lw-ppocr.js` SDK and offline HTML can be loaded directly from
`file://`. They normally run the single-threaded Emscripten module inside a
Blob Web Worker so inference does not
block the page UI; browsers that reject Blob Workers use a main-thread fallback.
The module does not use pthreads and retains one internal OCR line worker. Its
default `LW_WASM_SIMD128=ON` build requires a browser with WebAssembly SIMD
support. Configure with `-DLW_WASM_SIMD128=OFF` only when a scalar fallback
is needed for an older browser. See the [Browser JavaScript SDK](web-sdk.md)
and [standalone HTML usage](standalone-html.md).

The current repository contains a development-time converter, a deployable
pure-C loader/session planner, complete REC scalar kernels, and a public
recognize-only C API for decoded BGR8 pixels. Its graph executor, BGR
preprocessing, UTF-8 dictionary loader, and CTC decoder are correctness-tested.
The local Windows packages are development artifacts tested on the stated host;
they are not yet remote-CI or physical Windows 7 compatibility claims.
Windows 10 x86 success checks 32-bit build/ABI, memory, and scalar execution
assumptions; it does not establish Windows 7 compatibility.
The CI claims above apply to the pinned GitHub-hosted runner and browser
baselines. They do not imply validation on every Windows edition, Linux
distribution, browser, or physical machine.

The manually dispatched `customer-linux-architectures` workflow builds amd64
and ARM64 packages on native GitHub-hosted Ubuntu 22.04 runners. It also builds
an explicitly experimental LoongArch64 package inside a pinned Debian 13 image
under QEMU user emulation. Every job tests the installed package with real REC,
DET, and full OCR, checks the ELF machine type and dynamic dependencies, and
packages `BUILD-ENVIRONMENT.txt` with the archive. A green QEMU job proves that
the generic scalar path can build and execute in that emulated baseline; it is
not evidence of Loongson model compatibility, native performance, or a specific
customer distribution. Those claims require testing the exact archive on the
customer machine.
