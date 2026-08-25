# LWM v0 format draft

Status: **experimental and not frozen**.

LWM means LightWeight Model. Version 0 is designed only for compiling the exact
PP-OCRv6 tiny REC graph documented in `SUPPORTED_OPS_V0.md`.

## Invariants

- little-endian;
- fixed-width integers and IEEE-754 FP32;
- no pointers, `size_t`, `long`, native handles, or compiler-native structs;
- all references are file offsets or table indexes;
- every section starts at an 8-byte-aligned offset;
- a loader treats the entire file as untrusted input;
- one file is portable across Windows x86/x64, Linux x64, and Linux ARM64;
- unknown versions, flags, operators, or non-zero reserved fields are rejected.

## File layout

```text
header
tensor table
node table
operator parameter records
string table
weight blob
checksum trailer (decision pending)
```

The converter writes sections in this exact order. Runtime execution does not
depend on section adjacency; it uses validated offsets.

## Header candidate

The on-disk record will be serialized field-by-field. The following C-like
declaration documents fields; it is not permission to cast mapped bytes to a
native struct.

```c
struct lwm_v0_header_disk {
    uint8_t  magic[4];            /* "LWM0" */
    uint16_t format_major;        /* 0 */
    uint16_t format_minor;        /* 1 */
    uint32_t header_size;
    uint32_t flags;
    uint32_t tensor_count;
    uint32_t node_count;
    uint32_t input_count;
    uint32_t output_count;
    uint64_t tensor_offset;
    uint64_t node_offset;
    uint64_t param_offset;
    uint64_t string_offset;
    uint64_t weight_offset;
    uint64_t file_size;
    uint64_t workspace_size;
    uint64_t content_checksum;
};
```

`header_size` allows compatible extension during v0 experiments. The checksum
algorithm and coverage are intentionally undecided until the writer and corrupt
model tests are implemented; a value of zero must not silently disable integrity
checking in a production format.

## Tensor candidate

```c
#define LWM_V0_MAX_DIMS 8

struct lwm_v0_tensor_disk {
    uint32_t dtype;
    uint32_t rank;
    int32_t  dimensions[LWM_V0_MAX_DIMS];
    uint32_t flags;
    uint32_t reserved;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t workspace_offset;
    uint64_t workspace_size;
};
```

Only `f32`, `i32`, `i64`, and `u8` are candidate dtypes. `-1` is the only
dynamic-dimension marker considered for v0, and only the documented REC batch
and width positions may use it. Other negative values are invalid.

## Node candidate

```c
#define LWM_V0_MAX_NODE_INPUTS  8
#define LWM_V0_MAX_NODE_OUTPUTS 4

struct lwm_v0_node_disk {
    uint16_t op;
    uint16_t input_count;
    uint16_t output_count;
    uint16_t flags;
    uint32_t inputs[LWM_V0_MAX_NODE_INPUTS];
    uint32_t outputs[LWM_V0_MAX_NODE_OUTPUTS];
    uint64_t param_offset;
    uint32_t param_size;
    uint32_t reserved;
};
```

Operator parameters are individually versioned fixed-width records. The node
record does not contain a giant union.

## Required loader validation

Before exposing a model handle, the loader must validate:

1. magic, supported version, header size, flags, and reserved fields;
2. exact `file_size` agreement;
3. overflow-safe `count * record_size` for every table;
4. overflow-safe and aligned section ranges entirely inside the file;
5. non-overlapping fixed sections and permitted weight ranges;
6. tensor rank/dtype/dimensions and overflow-safe element/byte counts;
7. node input/output counts and every tensor index;
8. operator parameter offset, size, version, and constraints;
9. constant data range and workspace range;
10. workspace limit before allocation;
11. graph input/output indexes and supported dynamic-width propagation;
12. checksum after structural bounds make checksum reads safe.

Failure returns a stable error code. Library code must never call `abort` or
`exit` for malformed input.

## REC conversion decisions still open

- checksum algorithm and canonical coverage;
- string table retention policy for diagnostics;
- exact operator IDs and parameter record versions;
- whether all four REC BatchNormalization nodes are fused or two remain;
- workspace planning alignment and maximum supported tensor bytes;
- graph-level representation of the REC width expression.

These decisions require converter output and corrupt-model tests. Therefore LWM
v0 must not yet be described as ABI- or format-stable.
