# Platform matrix

Claims are deliberately separated into design, CI, and physical-machine
verification.

| Platform | Design target | CI verified | Physical machine verified | Current status |
|---|---:|---:|---:|---|
| Windows 10/11 x64 | Yes | No | Development host only | Primary implementation target |
| Linux x86_64 | Yes | No | No | Primary implementation target |
| Linux ARM64 | Yes | No | No | Planned primary target |
| Windows 7 SP1 x64 | Yes | No | No | Planned compatibility validation |
| Windows 7 SP1 x86 | Yes | No | No | Compatibility profile; not a v0.1 blocker |

The current repository contains a development-time ONNX analyzer, not a
deployable runtime. A green analyzer job does not establish runtime platform
support.
