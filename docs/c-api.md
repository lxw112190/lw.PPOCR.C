# Experimental C API

The public API in `include/lw_infer.h` is available for integration experiments
but is not ABI-frozen before 1.0. It currently covers model loading and REC
session planning. Complete REC graph execution exists behind a private test
interface but is not exposed in the public header yet.

## Ownership and thread model

- `lw_model_load` creates a read-only model handle; `lw_model_free` destroys it.
- `lw_session_create` creates an independent resolved tensor table and workspace.
- A model must outlive every session created from it.
- Separate sessions do not share mutable workspace and may later run in parallel.
- A single session is not promised to be safe for concurrent calls.
- Every successful create has one matching free; all free functions accept null.

Paths passed to the library are UTF-8. The Windows implementation converts to
UTF-16 before opening a file.

## REC session planning

The exact v0.1 REC model has one FP32 input. The descriptor must be:

```text
[N, 3, 48, W]
```

`N` and `W` are positive runtime values. A width that collapses below a valid
spatial output is rejected with `LW_STATUS_INVALID_SHAPE`.

Session creation performs all of the following before returning:

1. validates descriptor size, dtype, rank, dimensions, and unused fields;
2. propagates concrete shapes through all 161 nodes;
3. checks every resolved tensor byte multiplication and `max_tensor_size`;
4. computes tensor birth and last-use nodes;
5. assigns reusable 64-byte-aligned workspace ranges with first-fit allocation;
6. checks `max_workspace_size` and allocates one aligned workspace block.

Default limits are 256 MiB for one tensor and 512 MiB for a session workspace.
Passing zero for either initialized option keeps its default.

## Minimal planning example

```c
lw_model* model = NULL;
lw_session* session = NULL;
lw_error error;
lw_tensor_desc input;
lw_tensor_desc output;
lw_session_info info;

lw_error_init(&error);
if (lw_model_load("rec.lwm", NULL, &model, &error) != LW_STATUS_OK) {
    /* error.code and error.message are available here */
}

lw_tensor_desc_init(&input);
input.dtype = LW_DTYPE_F32;
input.rank = 4;
input.dimensions[0] = 1;
input.dimensions[1] = 3;
input.dimensions[2] = 48;
input.dimensions[3] = 320;

if (lw_session_create(model, &input, 1, NULL, &session, &error) == LW_STATUS_OK) {
    lw_session_info_init(&info);
    lw_tensor_desc_init(&output);
    lw_session_get_info(session, &info);
    lw_session_get_output_desc(session, 0, &output);
    /* output is [1, 40, 6906] for width 320 */
}

lw_session_free(session);
lw_model_free(model);
```

No public input data pointer is accepted yet, which prevents callers from
mistaking a successful plan for completed inference. The internal executor is
not an integration contract and may change without ABI notice.

## Errors

Programs should branch on `lw_status`, not parse diagnostic text. Relevant
planning errors are:

- `LW_STATUS_INVALID_ARGUMENT`: null pointer, wrong count, or bad options ABI;
- `LW_STATUS_INVALID_SHAPE`: incompatible dtype/rank/dimension/operator shape;
- `LW_STATUS_MEMORY_LIMIT`: tensor/workspace limit or size overflow;
- `LW_STATUS_OUT_OF_MEMORY`: host allocation failure;
- model load errors defined by the LWM loader.
