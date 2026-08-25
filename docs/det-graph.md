# DET probability graph

The exact bundled PP-OCRv6 tiny DET ONNX model is deterministically converted
to LWM v0.1 and executed by the private pure-C runtime. For an FP32 NCHW input
`[1,3,H,W]`, the graph resolves and produces `[1,1,H,W]` probabilities.

## Exact supported transformations

- eight GlobalAveragePool nodes become equivalent ReduceMean nodes;
- six constant-scale Resize nodes become nearest/asymmetric/floor LWM Resize;
- exact unit-stride `SAME_UPPER` Conv and MaxPool padding becomes explicit;
- unused shape/scale constants are removed;
- all other data-path nodes and FP32 weights retain their graph order.

Conversion is guarded by the source model SHA-256. Unknown models, attributes,
Resize coordinate modes, nearest modes, or dynamic scale inputs are rejected.

## Correctness gates

Direct NumPy/ONNX reference tests cover Sigmoid, Concat, nearest Resize,
MaxPool, and ConvTranspose. The complete graph is run twice for determinism at
`32x32` and `32x64`, and every output value is compared with ONNX Runtime. The
x64 development result has maximum absolute error below `1.3e-7`.

## Deliberate boundary

This milestone ends at the probability map. It does not define decoded-image
preprocessing, resize/padding restoration, thresholding, dilation, contours,
DB unclip, polygon/quad selection, scores, coordinate mapping, or a public
detector ABI. Those contracts must be implemented and reference-tested before
applications receive text boxes.
