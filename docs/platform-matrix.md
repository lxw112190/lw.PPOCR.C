# Platform matrix

Claims are deliberately separated into design, CI, and physical-machine
verification.

| Platform | Design target | CI verified | Physical machine verified | Current status |
|---|---:|---:|---:|---|
| Windows 10/11 x64 | Yes | No | Windows 10 development host | Full REC path, DLL/static package, and installed Demo pass |
| Linux x86_64 | Yes | No | No | Primary implementation target |
| Linux ARM64 | Yes | No | No | Planned primary target |
| Windows 7 SP1 x64 | Yes | No | No | Planned compatibility validation |
| Windows 7 SP1 x86 | Yes | No | No | Compatibility profile; not a v0.1 blocker |
| Windows 10 x86 build | Design check | No | Windows 10 development host | Full REC path, DLL/static package, and installed Demo pass |

The current repository contains a development-time converter, a deployable
pure-C loader/session planner, complete REC scalar kernels, and a public
recognize-only C API for decoded BGR8 pixels. Its graph executor, BGR
preprocessing, UTF-8 dictionary loader, and CTC decoder are correctness-tested.
The local Windows packages are development artifacts tested on the stated host;
they are not yet remote-CI or physical Windows 7 compatibility claims.
Windows 10 x86 success checks 32-bit build/ABI, memory, and scalar execution
assumptions; it does not establish Windows 7 compatibility.
The CI column remains `No` until the updated workflow has run in the remote
repository.
