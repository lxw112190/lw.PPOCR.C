# LWM v0.1 format

Status: **experimental and not frozen**.

LWM means LightWeight Model. Version 0.1 currently encodes only the exact
PP-OCRv6 tiny REC graph documented in `SUPPORTED_OPS_V0.md`. It is not an ONNX
container and the runtime is not a general ONNX executor.

## Encoding invariants

- little-endian fixed-width integers and IEEE-754 FP32;
- no pointer, `size_t`, `long`, native handle, or compiler-native struct;
- every section starts at an 8-byte-aligned absolute file offset;
- records are serialized and read field-by-field, never by casting file bytes;
- unknown versions, flags, operator IDs, parameter versions, and non-zero
  reserved fields are rejected;
- the same file is intended for Windows x86/x64 and Linux x64/ARM64;
- all model bytes are untrusted and must pass validation before a model handle
  is returned.

## Canonical layout

```text
160-byte header
graph input tensor-index table (u32[])
graph output tensor-index table (u32[])
80-byte tensor records
72-byte node records
operator parameter records
string table (empty in v0.1)
FP32 weight blob
```

Padding bytes are zero. The writer produces sections in this exact order. A
loader uses validated offsets and sizes rather than assuming adjacency.

## Header: 160 bytes

| Offset | Type | Field |
|---:|---|---|
| 0 | `u8[4]` | magic, `LWM0` |
| 4 | `u16` | format major, `0` |
| 6 | `u16` | format minor, `1` |
| 8 | `u32` | header size, `160` |
| 12 | `u32` | flags |
| 16 | `u32` | tensor count |
| 20 | `u32` | node count |
| 24 | `u32` | graph input count |
| 28 | `u32` | graph output count |
| 32 | `u64` | graph input table offset |
| 40 | `u64` | graph output table offset |
| 48 | `u64` | tensor table offset |
| 56 | `u64` | node table offset |
| 64 | `u64` | parameter section offset |
| 72 | `u64` | parameter section size |
| 80 | `u64` | string section offset |
| 88 | `u64` | string section size |
| 96 | `u64` | weight section offset |
| 104 | `u64` | weight section size |
| 112 | `u64` | exact file size |
| 120 | `u64` | workspace size |
| 128 | `u64` | content checksum |
| 136 | `u64[3]` | reserved, all zero |

The only v0.1 header flag is bit 0, `NO_MEMORY_PLAN`. It must be set and
`workspace_size` must be zero. A later experimental revision will replace this
with a converter-generated lifetime/workspace plan.

## Tensor record: 80 bytes

| Offset | Type | Field |
|---:|---|---|
| 0 | `u32` | dtype (`1=f32`, `2=i32`, `3=i64`, `4=u8`) |
| 4 | `u32` | rank, at most 8 |
| 8 | `i32[8]` | dimensions; unused slots are zero |
| 40 | `u32` | flags: constant/input/output |
| 44 | `u32` | reserved, zero |
| 48 | `u64` | absolute constant-data offset, otherwise zero |
| 56 | `u64` | constant-data byte size, otherwise zero |
| 64 | `u64` | workspace offset; `UINT64_MAX` in v0.1 |
| 72 | `u64` | workspace byte size; zero in v0.1 |

`-1` is the only dynamic-dimension marker. Constant tensors cannot contain
dynamic dimensions, and their byte size must exactly equal the overflow-checked
shape product times the dtype size.

## Node record: 72 bytes

| Offset | Type | Field |
|---:|---|---|
| 0 | `u16` | operator ID |
| 2 | `u16` | input count, at most 8 |
| 4 | `u16` | output count, 1 through 4 |
| 6 | `u16` | flags, zero |
| 8 | `u32[8]` | input tensor indexes; unused slots zero |
| 40 | `u32[4]` | output tensor indexes; unused slots zero |
| 56 | `u64` | absolute parameter offset, or zero |
| 64 | `u32` | exact parameter record size, or zero |
| 68 | `u32` | reserved, zero |

## Operator IDs and parameter records

Every non-empty parameter record begins with `u16 version=1`. Records use only
fixed-width fields and have an exact size checked by the loader.

| ID | Operator | Parameter bytes |
|---:|---|---:|
| 1 | Conv | 64 |
| 2 | Add | 0 |
| 3 | Mul | 0 |
| 4 | Div | 0 |
| 5 | Erf | 0 |
| 6 | HardSigmoid | 16 |
| 7 | BatchNormalization | 24 |
| 8 | ReduceMean | 48 |
| 9 | Relu | 0 |
| 10 | AveragePool | 64 |
| 11 | Squeeze | 40 |
| 12 | Transpose | 40 |
| 13 | Unsqueeze | 40 |
| 14 | MatMul | 0 |
| 15 | Softmax | 16 |

The writer normalizes omitted ONNX attributes to opset-11 defaults before
encoding them. Conv and pooling records contain rank, kernel, stride, dilation,
padding, group, and applicable boolean attributes. Axis-based records contain a
count followed by eight signed axis/perm slots. All unused and reserved fields
are zero.

## Checksum

`content_checksum` is FNV-1a 64 over the complete file while header bytes
128..135 are treated as zero. The stored checksum must be non-zero. FNV-1a is a
small deterministic corruption check, not a cryptographic signature; model
authenticity must be handled by a trusted distribution channel and published
SHA-256 hashes.

## Current REC conversion policy

- requires the bundled REC SHA-256 identity, one input, one output, and default
  ONNX opset 11;
- rejects every operator outside the 15 IDs above;
- removes 58 one-input/one-output Identity aliases;
- emits 161 executable nodes and 282 tensors;
- retains all four BatchNormalization nodes;
- keeps canonical little-endian FP32 weights;
- does not fuse operators, pack weights, retain names, or create a workspace
  plan yet.

These choices deliberately avoid numerical graph rewrites before golden
reference tests exist.

## Required loader validation

Before exposing a model handle, the loader validates magic/version/header,
exact file size, flags/reserved fields, aligned overflow-safe section ranges,
section ordering and non-overlap, checksum, tensor dtype/rank/dimensions/data,
node arity/operator/indexes, exact parameter size/version/constraints, graph
input/output indexes and flags, and the no-workspace-plan invariant.

Any malformed file returns a stable `lw_status`; library code never calls
`abort` or `exit`.
