# Model assets

The repository currently includes only the platform-independent PP-OCRv6 tiny
ONNX models, UTF-8 dictionary, and sample image needed for converter analysis.
Deployment-specific TensorRT engines are intentionally excluded.

The ONNX files are converter inputs. They will not be parsed by or distributed
as dependencies of the future pure-C runtime package.
