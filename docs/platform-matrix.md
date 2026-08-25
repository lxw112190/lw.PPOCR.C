# Platform matrix

Claims are deliberately separated into design, CI, and physical-machine
verification.

| Platform | Design target | CI verified | Physical machine verified | Current status |
|---|---:|---:|---:|---|
| Windows 10/11 x64 | Yes | No | Windows 10 development host | Converter/loader/tests pass locally |
| Linux x86_64 | Yes | No | No | Primary implementation target |
| Linux ARM64 | Yes | No | No | Planned primary target |
| Windows 7 SP1 x64 | Yes | No | No | Planned compatibility validation |
| Windows 7 SP1 x86 | Yes | No | No | Compatibility profile; not a v0.1 blocker |
| Windows 10 x86 build | Design check | No | Windows 10 development host | Converter/loader/ABI/tests pass locally |

The current repository contains a development-time converter and a deployable
pure-C loader, but no inference executor. Windows 10 x86 success checks 32-bit
build/ABI assumptions; it does not establish Windows 7 compatibility. The CI
column remains `No` until the updated workflow has run in the remote repository.
