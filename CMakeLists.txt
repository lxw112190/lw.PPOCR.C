cmake_minimum_required(VERSION 3.20)

project(lw.PPOCR.C VERSION 1.2.2 LANGUAGES C)

option(LW_BUILD_HTTP_DEMO
    "Build the cross-platform cpp-httplib HTTP OCR Demo" ON)
option(LW_BUILD_CSHARP_DEMOS
    "Build the .NET Framework 3.5 WinForms Demo on Windows" OFF)
option(LW_RUNTIME_ONLY
    "Build only the dependency-free native runtime for an embedding target" OFF)
option(LW_DEPLOY_ONNX_MODELS
    "Deploy official ONNX assets in Web/Node packages (LWM remains supported)" ON)
option(LW_WASM_SIMD128
    "Enable portable WebAssembly SIMD128 for the standalone OCR HTML" ON)
option(LW_WASM_COMPILED_REC
    "Experimentally run WASM REC on the compiled physical backend" OFF)
option(LW_WASM_FINE_REC_WIDTHS
    "Use 64-pixel adaptive REC width buckets in compiled WASM OCR" OFF)
option(LW_ARM64_FINE_REC_WIDTHS
    "Experimentally reduce ARM64 REC padding with finer width buckets" OFF)
option(LW_NATIVE_FINE_REC_WIDTHS
    "Experimentally reduce native compiled REC padding with finer widths (including 480)" OFF)
option(LW_EXPERIMENTAL_REC_FFN_TILING
    "Experimental cache-local Pointwise/GELU/Pointwise scheduling (no arithmetic changes)" OFF)
option(LW_EXPERIMENTAL_REC_FFN_FUSION
    "Experimental native REC FFN with tile-local hidden storage" OFF)
option(LW_EXPERIMENTAL_REC_DW_PW_FUSION
    "Experimental native REC depthwise-to-pointwise tile fusion" OFF)
set(LW_REC_DW_PW_TILE_PIXELS "48" CACHE STRING "DW/PW pixel tile (12, 48 or 96)")
set_property(CACHE LW_REC_DW_PW_TILE_PIXELS PROPERTY STRINGS 12 48 96)
if(NOT LW_REC_DW_PW_TILE_PIXELS MATCHES "^(12|48|96)$")
    message(FATAL_ERROR "LW_REC_DW_PW_TILE_PIXELS must be 12, 48 or 96")
endif()
set(LW_REC_FFN_TILE_PIXELS "96" CACHE STRING "Experimental FFN pixel block (12, 48 or 96)")
set_property(CACHE LW_REC_FFN_TILE_PIXELS PROPERTY STRINGS 12 48 96)
if(NOT LW_REC_FFN_TILE_PIXELS MATCHES "^(12|48|96)$")
    message(FATAL_ERROR "LW_REC_FFN_TILE_PIXELS must be 12, 48 or 96")
endif()
option(LW_WASM_COMPILED_DET
    "Run WASM DET on the compiled NHWC backend when compiled REC is enabled" ON)
option(LW_WASM_REC_SIMD_KERNELS
    "Use SIMD128 Dense and Depthwise kernels in experimental WASM compiled REC" ON)
set(LW_WASM_POINTWISE_ROWS "2" CACHE STRING
    "WASM SIMD128 pointwise pixel tile; supported values: 2 or 4")
set_property(CACHE LW_WASM_POINTWISE_ROWS PROPERTY STRINGS 2 4)
if(NOT LW_WASM_POINTWISE_ROWS STREQUAL "2" AND
   NOT LW_WASM_POINTWISE_ROWS STREQUAL "4")
    message(FATAL_ERROR "LW_WASM_POINTWISE_ROWS must be 2 or 4")
endif()
option(LW_WASM_DB_BITMAP_SIMD
    "Experimentally use SIMD128 DB thresholding in compiled WASM OCR" OFF)
option(LW_WEB_LEGACY
    "Build the Tiny scalar Web compatibility distribution" OFF)
set(LW_WEB_MIN_CHROME_VERSION "70" CACHE STRING
    "Minimum Chromium version for the legacy browser distribution")
option(LW_REC_RESIDENT_WIDTHS
    "Prewarm all standard REC widths for native performance experiments" OFF)
option(LW_EXPERIMENTAL_AVX2_FMA_DISPATCH
    "Use experimental AVX2+FMA Conv1x1 and terminal MatMul paths in native builds" OFF)
option(LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH
    "Use the experimental AVX2+FMA Conv1x1 dispatch in native builds" OFF)
option(LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH
    "Use the experimental AVX2+FMA terminal MatMul dispatch in native builds" OFF)
option(LW_EXPERIMENTAL_AVX2_FAST_PATH
    "Enable compiled AVX2/FMA x64 REC backends for adaptive widths" ON)
option(LW_EXPERIMENTAL_ADAPTIVE_REC_BUDGET
    "Share the native x64 and ARM64 REC thread budget across active line workers" ON)
set(LW_X64_OCR_THREAD_BUDGET "8" CACHE STRING
    "Native x64 OCR thread budget (8: physical-core policy, 16: logical-processor policy)")
set_property(CACHE LW_X64_OCR_THREAD_BUDGET PROPERTY STRINGS 8 16)
if(NOT LW_X64_OCR_THREAD_BUDGET MATCHES "^(8|16)$")
    message(FATAL_ERROR "LW_X64_OCR_THREAD_BUDGET must be 8 or 16")
endif()
option(LW_EXPERIMENTAL_DET_FIXED_POINT_RESIZE
    "Use the OpenCV-parity fixed-point DET resize with LUT normalization" ON)
option(LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE
    "Use the keep-aspect fixed-point CLS resize with -1 padding" ON)
option(LW_EXPERIMENTAL_PERSISTENT_LINE_POOL
    "Reuse a persistent line-worker thread pool across OCR requests" ON)
option(LW_EXPERIMENTAL_PARALLEL_CROP_PHASE
    "Rasterize perspective crops as parallel row-band jobs" ON)
option(LW_EXPERIMENTAL_PREPARED_EXECUTION
    "Build the experimental session-local prepared execution table" OFF)
option(LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2
    "Use the experimental size-descending workspace interval planner" OFF)
option(LW_EXPERIMENTAL_FUSION_MEMORY
    "Use the experimental physical lifetime plan for fused GELU" OFF)
option(LW_EXPERIMENTAL_CTC_TILED
    "Use the experimental physical plan for greedy CTC output elision" OFF)
option(LW_WASM_REC_LAZY_FALLBACK
    "Avoid canonical REC execution allocations until a compiled WASM fallback is needed" OFF)
option(LW_EXPERIMENTAL_STREAMING_CROPS
    "Use experimental worker-local streaming crop buffers" OFF)
option(LW_AVX2_FMA_CONV3X3_DISPATCH
    "Enable the measured AVX2+FMA Conv3x3 Node 6 dispatch" ON)
option(LW_EXPERIMENTAL_AVX2_CONV3X3_EIGHT_OUTPUTS
    "Use experimental AVX2 eight-output blocking for unit-stride Conv3x3" OFF)
set(LW_WASM_HOST_ABI_VERSION "1" CACHE STRING
    "WASM Host ABI version embedded in the Web adapter and Node manifest")
set(LW_LWM_FORMAT_VERSION "0.1" CACHE STRING
    "LWM model format version recorded in distribution manifests")
option(LW_WEB_PDF
    "Embed offline PDF.js support in the standalone OCR HTML" ON)
option(LW_BUILD_WEB_MODEL_VARIANTS
    "Build standalone PP-OCRv6 Small and Medium browser SDK/HTML variants" OFF)
set(LW_PRECONVERTED_ASSETS_DIR "" CACHE PATH
    "Directory containing preconverted LWM models and PPM fixtures")
set(LW_BUILD_ENVIRONMENT_FILE "" CACHE FILEPATH
    "Optional build-environment report installed in the package root")
set(LW_STAGED_PACKAGE_TEST_TIMEOUT 240 CACHE STRING
    "Timeout in seconds for the installed-package smoke test")
set(LW_HTTP_TEST_REQUEST_TIMEOUT 30 CACHE STRING
    "Timeout in seconds for one HTTP package-test request")
set(LW_HTTP_TEST_PROCESS_TIMEOUT 120 CACHE STRING
    "Timeout in seconds for the complete HTTP package-test process")
set(LW_PACKAGE_PLATFORM_OVERRIDE "" CACHE STRING
    "Optional explicit platform suffix for the CPack archive name")

if(LW_BUILD_HTTP_DEMO)
    enable_language(CXX)
endif()

if(NOT LW_WASM_HOST_ABI_VERSION MATCHES "^[0-9]+$")
    message(FATAL_ERROR "LW_WASM_HOST_ABI_VERSION must be a positive integer")
endif()
if(NOT LW_LWM_FORMAT_VERSION MATCHES "^[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "LW_LWM_FORMAT_VERSION must use the form major.minor")
endif()
if(LW_EXPERIMENTAL_FUSION_MEMORY AND NOT LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2)
    message(FATAL_ERROR
        "LW_EXPERIMENTAL_FUSION_MEMORY requires LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2")
endif()
if(LW_EXPERIMENTAL_CTC_TILED AND NOT LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2)
    message(FATAL_ERROR
        "LW_EXPERIMENTAL_CTC_TILED requires LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2")
endif()

if(LW_WEB_LEGACY)
    if(LW_WASM_SIMD128)
        message(FATAL_ERROR
            "LW_WEB_LEGACY requires LW_WASM_SIMD128=OFF")
    endif()
    if(LW_WEB_PDF)
        message(FATAL_ERROR
            "LW_WEB_LEGACY requires LW_WEB_PDF=OFF")
    endif()
    if(LW_BUILD_WEB_MODEL_VARIANTS)
        message(FATAL_ERROR
            "LW_WEB_LEGACY supports PP-OCRv6 Tiny only")
    endif()
    if(NOT LW_WEB_MIN_CHROME_VERSION MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "LW_WEB_MIN_CHROME_VERSION must be an integer")
    endif()
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckCCompilerFlag)
if(NOT LW_RUNTIME_ONLY)
    find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
endif()
if(NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
endif()

include(cmake/ModelAssets.cmake)
include(cmake/RuntimeTargets.cmake)

if(LW_RUNTIME_ONLY)
    message(STATUS "LW_RUNTIME_ONLY enabled: skipping model conversion, demos, tests, install and packaging")
    return()
endif()

include(cmake/WebTargets.cmake)
include(cmake/DemoTargets.cmake)
include(cmake/NativeTests.cmake)
include(cmake/Packaging.cmake)
