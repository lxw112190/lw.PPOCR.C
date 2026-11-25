# Runtime sources, ISA compile options, and static/shared library targets.

set(LW_PPOCR_C_SOURCES
    src/kernels/packed_conv.c
    src/kernels/packed_conv3x3.c
    src/kernels/packed_matmul.c
    src/kernels/scalar_activation.c
    src/kernels/scalar_conv.c
    src/kernels/scalar_elementwise.c
    src/kernels/scalar_layout.c
    src/kernels/scalar_matmul.c
    src/kernels/scalar_reduction.c
    src/kernels/scalar_softmax.c
    src/ppocr/classifier.c
    src/ppocr/cls_preprocess.c
    src/ppocr/ctc_decode.c
    src/ppocr/crop.c
    src/ppocr/crop_buffer.c
    src/ppocr/db_geometry.c
    src/ppocr/db_postprocess.c
    src/ppocr/det_preprocess.c
    src/ppocr/det_preprocess_fixed.c
    src/ppocr/detector.c
    src/ppocr/ocr.c
    src/ppocr/profile.c
    src/ppocr/reading_order.c
    src/ppocr/rec_preprocess.c
    src/ppocr/recognizer.c
    src/runtime/executor.c
    src/runtime/cpu_topology.c
    src/runtime/execution_plan.c
    src/runtime/memory.c
    src/runtime/model.c
    src/runtime/onnx_import.c
    src/runtime/parallel.c
    src/runtime/ctc_head_parallel_internal.c
    src/runtime/prepared_execution.c
    src/runtime/session.c
    src/runtime/shape.c
    src/runtime/validate.c
    src/simd/avx2_binary.c
    src/simd/avx2_bitmap.c
    src/simd/avx2_conv1x1.c
    src/simd/avx2_conv2x2.c
    src/simd/avx2_conv3x3.c
    src/simd/avx2_conv3x3_eight.c
    src/simd/avx2_conv7x7.c
    src/simd/avx2_convtranspose2x2.c
    src/simd/avx2_depthwise3x3.c
    src/simd/avx2_depthwise5x5.c
    src/simd/avx2_depthwise9x9.c
    src/simd/avx2_erf.c
    src/simd/avx2_transpose.c
    src/simd/avx2_softmax.c
    src/simd/avx2_matmul.c
    src/simd/avx2_fma_packed_matmul.c
    src/simd/avx2_packed_conv1x1.c
    src/simd/avx2_fma_packed_conv1x1.c
    src/simd/avx2_fma_packed_conv1x1_8x8.c
    src/simd/cpu_features.c
    src/simd/lsx_packed_conv1x1.c
    src/simd/neon_conv3x3.c
    src/simd/neon_convtranspose2x2.c
    src/simd/neon_depthwise5x5.c
    src/simd/neon_matmul.c
    src/simd/neon_packed_conv1x1.c
    src/simd/sse2_binary.c
    src/simd/sse2_conv1x1.c
    src/simd/sse2_conv2x2.c
    src/simd/sse2_conv3x3.c
    src/simd/sse2_convtranspose2x2.c
    src/simd/sse2_depthwise3x3.c
    src/simd/sse2_depthwise5x5.c
    src/simd/sse2_matmul.c
    src/simd/sse2_packed_conv1x1.c
    src/simd/wasm128_erf.c)

set(LW_X64_REC_FAST_ENABLED OFF)
set(LW_WASM_COMPILED_REC_ENABLED OFF)
if(LW_EXPERIMENTAL_AVX2_FAST_PATH AND NOT EMSCRIPTEN AND
   (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x64|X86)$" OR
    MSVC_C_ARCHITECTURE_ID MATCHES "^(x64|X86)$"))
    set(LW_X64_REC_FAST_ENABLED ON)
endif()
if(EMSCRIPTEN AND LW_WASM_SIMD128 AND LW_WASM_COMPILED_REC)
    set(LW_WASM_COMPILED_REC_ENABLED ON)
elseif(EMSCRIPTEN AND LW_WASM_COMPILED_REC)
    message(FATAL_ERROR "LW_WASM_COMPILED_REC requires LW_WASM_SIMD128=ON")
endif()
if(LW_WASM_REC_LAZY_FALLBACK AND
   (NOT LW_WASM_COMPILED_REC_ENABLED OR LW_EXPERIMENTAL_CTC_TILED))
    message(FATAL_ERROR
        "LW_WASM_REC_LAZY_FALLBACK requires compiled SIMD128 REC without CTC tiled mode")
endif()
if(LW_WASM_FINE_REC_WIDTHS AND NOT LW_WASM_COMPILED_REC_ENABLED)
    message(FATAL_ERROR "LW_WASM_FINE_REC_WIDTHS requires compiled SIMD128 WASM REC")
endif()
if(LW_NATIVE_FINE_REC_WIDTHS AND
   (NOT LW_X64_REC_FAST_ENABLED OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8))
    message(FATAL_ERROR "LW_NATIVE_FINE_REC_WIDTHS requires compiled native x64 REC")
endif()
if(LW_EXPERIMENTAL_REC_DW_PW_FUSION AND
   (NOT LW_X64_REC_FAST_ENABLED OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR EMSCRIPTEN OR
    LW_EXPERIMENTAL_REC_FFN_TILING OR LW_EXPERIMENTAL_REC_FFN_FUSION))
    message(FATAL_ERROR "DW/PW fusion requires native x64 REC with FFN experiments OFF")
endif()
if(LW_X64_REC_FAST_ENABLED OR LW_WASM_COMPILED_REC_ENABLED)
    list(APPEND LW_PPOCR_C_SOURCES
        src/runtime/layout_planner.c
        src/runtime/x64_rec_fast_plan.c
        src/runtime/x64_rec_fast_executor.c
        src/runtime/x64_rec_backend_compile.c
        src/runtime/x64_rec_backend_execute.c
        src/runtime/x64_rec_backend_arena.c
        src/runtime/rec_backend_kernels.c
        src/runtime/x64_det_backend_compile.c
        src/runtime/x64_det_backend_execute.c
        src/runtime/ctc_projection_internal.c
        src/runtime/layout_convert_x64.c
        src/kernels/nhwc_pack.c
        src/kernels/nhwc_depthwise.c
        src/simd/avx2_nhwc_pointwise.c
        src/simd/avx2_nhwc_depthwise.c
        src/simd/avx2_nhwc_support.c
        src/simd/avx2_nhwc_dense.c
        src/simd/avx2_nhwc_convtranspose.c)
endif()
if(LW_WASM_COMPILED_REC_ENABLED)
    list(APPEND LW_PPOCR_C_SOURCES
        src/simd/wasm128_nhwc_pointwise.c
        src/simd/wasm128_elementwise.c
        src/simd/wasm128_matmul.c
        src/simd/wasm128_nhwc_convtranspose.c
        src/simd/wasm128_packed_matmul.c)
    if(LW_WASM_DB_BITMAP_SIMD)
        list(APPEND LW_PPOCR_C_SOURCES src/simd/wasm128_bitmap.c)
    endif()
endif()
if(MSVC AND MSVC_C_ARCHITECTURE_ID MATCHES "^(x64|X86)$")
    set_property(SOURCE src/simd/avx2_binary.c src/simd/avx2_conv1x1.c
        src/simd/avx2_conv2x2.c
        src/simd/avx2_conv3x3.c src/simd/avx2_conv3x3_eight.c
        src/simd/avx2_conv7x7.c
        src/simd/avx2_convtranspose2x2.c
        src/simd/avx2_depthwise3x3.c
        src/simd/avx2_depthwise5x5.c
        src/simd/avx2_depthwise9x9.c
        src/simd/avx2_erf.c src/simd/avx2_transpose.c src/simd/avx2_matmul.c
        src/simd/avx2_fma_packed_matmul.c
        src/simd/avx2_packed_conv1x1.c src/simd/avx2_fma_packed_conv1x1.c
        src/simd/avx2_fma_packed_conv1x1_8x8.c APPEND PROPERTY
        COMPILE_OPTIONS /arch:AVX2 /fp:strict)
    set_property(SOURCE src/simd/cpu_features.c APPEND PROPERTY
        COMPILE_OPTIONS /arch:AVX)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LW_SYSTEM_PROCESSOR_LOWER)
set(LW_ADAPTIVE_REC_BUDGET_ENABLED OFF)
if(LW_EXPERIMENTAL_ADAPTIVE_REC_BUDGET AND NOT EMSCRIPTEN AND
   (LW_X64_REC_FAST_ENABLED OR LW_SYSTEM_PROCESSOR_LOWER MATCHES "^(arm64|aarch64)$"))
    set(LW_ADAPTIVE_REC_BUDGET_ENABLED ON)
endif()

# Keep the portable Conv references and their SIMD implementations on the same
# multiply-then-add contract. GCC enables FP contraction by default on AArch64
# and LoongArch, while the intrinsic kernels below deliberately use separate
# multiply and add instructions so their outputs remain byte-identical to the
# established x86 kernels and golden checksums.
set(LW_REPRODUCIBLE_CONV_SOURCES
    src/kernels/packed_conv.c
    src/kernels/packed_conv3x3.c
    src/kernels/scalar_conv.c
    src/simd/lsx_packed_conv1x1.c
    src/simd/neon_conv3x3.c
    src/simd/neon_convtranspose2x2.c
    src/simd/neon_depthwise5x5.c
    src/simd/neon_packed_conv1x1.c)

if(MSVC AND LW_SYSTEM_PROCESSOR_LOWER MATCHES "^(arm64|aarch64)$")
    set_property(SOURCE ${LW_REPRODUCIBLE_CONV_SOURCES} APPEND PROPERTY
        COMPILE_OPTIONS /fp:strict)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set_property(SOURCE ${LW_REPRODUCIBLE_CONV_SOURCES} APPEND PROPERTY
        COMPILE_OPTIONS -ffp-contract=off)
endif()

if(LW_SYSTEM_PROCESSOR_LOWER MATCHES "^(loongarch64|loong64)$")
    check_c_compiler_flag("-mlsx" LW_COMPILER_SUPPORTS_LSX)
    if(NOT LW_COMPILER_SUPPORTS_LSX)
        message(FATAL_ERROR "The LoongArch compiler must support -mlsx")
    endif()
    set_property(SOURCE src/simd/lsx_packed_conv1x1.c APPEND PROPERTY
        COMPILE_OPTIONS -mlsx -mno-lasx)
endif()

function(lw_configure_runtime_target target)
    if(LW_EXPERIMENTAL_REC_DW_PW_FUSION AND LW_X64_REC_FAST_ENABLED AND NOT EMSCRIPTEN)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_REC_DW_PW_FUSION=1
            LW_REC_DW_PW_TILE_PIXELS=${LW_REC_DW_PW_TILE_PIXELS})
    endif()
    if(LW_EXPERIMENTAL_REC_FFN_FUSION AND LW_X64_REC_FAST_ENABLED AND NOT EMSCRIPTEN)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_REC_FFN_FUSION=1
            LW_REC_FFN_TILE_PIXELS=${LW_REC_FFN_TILE_PIXELS})
    endif()
    if(LW_EXPERIMENTAL_REC_FFN_TILING)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_REC_FFN_TILING=1
            LW_REC_FFN_TILE_PIXELS=${LW_REC_FFN_TILE_PIXELS})
    endif()
    if(LW_NATIVE_FINE_REC_WIDTHS AND LW_X64_REC_FAST_ENABLED)
        target_compile_definitions(${target} PRIVATE LW_NATIVE_FINE_REC_WIDTHS=1)
    endif()
    target_include_directories(${target}
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(${target} PUBLIC c_std_11)
    if(NOT EMSCRIPTEN)
        target_link_libraries(${target} PUBLIC Threads::Threads)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_link_libraries(${target} PUBLIC m)
    endif()
    if(EMSCRIPTEN AND LW_WASM_SIMD128)
        # Keep the offline single-HTML build single-threaded, but compile its
        # scalar/SSE2-compatible kernels to portable WebAssembly SIMD128.
        # -msse2 enables Emscripten's compatible <emmintrin.h> surface;
        # -msimd128 lowers those intrinsics to portable WASM SIMD instructions.
        target_compile_options(${target} PRIVATE -O3 -msimd128 -msse2)
    endif()
endfunction()

function(lw_configure_runtime_features target)
    if(LW_ADAPTIVE_REC_BUDGET_ENABLED)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_ADAPTIVE_REC_BUDGET=1)
    endif()
    if(LW_EXPERIMENTAL_DET_FIXED_POINT_RESIZE)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_DET_FIXED_POINT_RESIZE=1)
    endif()
    if(LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE=1)
    endif()
    if(LW_EXPERIMENTAL_PERSISTENT_LINE_POOL)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_PERSISTENT_LINE_POOL=1)
    endif()
    if(LW_EXPERIMENTAL_PARALLEL_CROP_PHASE)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_PARALLEL_CROP_PHASE=1)
    endif()
    if(LW_REC_RESIDENT_WIDTHS AND NOT EMSCRIPTEN)
        target_compile_definitions(${target} PRIVATE LW_REC_RESIDENT_WIDTHS=1)
    endif()
    if((LW_EXPERIMENTAL_AVX2_FMA_DISPATCH OR LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH) AND NOT EMSCRIPTEN)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH=1)
    endif()
    if((LW_EXPERIMENTAL_AVX2_FMA_DISPATCH OR LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH) AND NOT EMSCRIPTEN)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH=1)
    endif()
    if(LW_AVX2_FMA_CONV3X3_DISPATCH AND NOT EMSCRIPTEN)
        target_compile_definitions(${target} PRIVATE LW_AVX2_FMA_CONV3X3_DISPATCH=1)
    endif()
    if(LW_EXPERIMENTAL_AVX2_CONV3X3_EIGHT_OUTPUTS AND NOT EMSCRIPTEN)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_AVX2_CONV3X3_EIGHT_OUTPUTS=1)
    endif()
    if(LW_EXPERIMENTAL_PREPARED_EXECUTION)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_PREPARED_EXECUTION=1)
    endif()
    if(LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2=1)
    endif()
    if(LW_EXPERIMENTAL_FUSION_MEMORY)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_FUSION_MEMORY=1)
    endif()
    if(LW_EXPERIMENTAL_CTC_TILED)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_CTC_TILED=1)
    endif()
    if(LW_EXPERIMENTAL_STREAMING_CROPS)
        target_compile_definitions(${target} PRIVATE LW_EXPERIMENTAL_STREAMING_CROPS=1)
    endif()
endfunction()

add_library(lw_ppocr_c STATIC ${LW_PPOCR_C_SOURCES})
set_target_properties(lw_ppocr_c PROPERTIES
    EXPORT_NAME static
    OUTPUT_NAME lw_ppocr_c_static
    POSITION_INDEPENDENT_CODE ON)
lw_configure_runtime_target(lw_ppocr_c)
if(LW_X64_REC_FAST_ENABLED OR LW_WASM_COMPILED_REC_ENABLED)
    target_compile_definitions(lw_ppocr_c PRIVATE LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
    if(LW_WASM_COMPILED_REC_ENABLED)
        if(LW_WASM_FINE_REC_WIDTHS)
            target_compile_definitions(lw_ppocr_c PRIVATE LW_WASM_FINE_REC_WIDTHS=1)
        endif()
        target_compile_definitions(lw_ppocr_c PRIVATE
            LW_WASM_POINTWISE_ROWS=${LW_WASM_POINTWISE_ROWS})
    endif()
    if(LW_WASM_COMPILED_REC_ENABLED AND LW_WASM_REC_SIMD_KERNELS)
        target_compile_definitions(lw_ppocr_c PRIVATE LW_WASM_REC_SIMD_KERNELS=1)
    endif()
    if(LW_WASM_COMPILED_REC_ENABLED AND LW_WASM_DB_BITMAP_SIMD)
        target_compile_definitions(lw_ppocr_c PRIVATE LW_WASM_DB_BITMAP_SIMD=1)
    endif()
    if(LW_WASM_COMPILED_REC_ENABLED AND LW_WASM_COMPILED_DET AND
       LW_EXPERIMENTAL_DET_FIXED_POINT_RESIZE)
        target_compile_definitions(lw_ppocr_c PRIVATE LW_WASM_COMPILED_DET=1)
    endif()
    if(MSVC)
        set_property(SOURCE src/simd/avx2_nhwc_pointwise.c src/simd/avx2_nhwc_dense.c src/simd/avx2_nhwc_depthwise.c src/simd/avx2_nhwc_support.c src/simd/avx2_nhwc_convtranspose.c APPEND PROPERTY
            COMPILE_OPTIONS /arch:AVX2 /fp:strict)
    endif()
endif()
if(LW_WASM_REC_LAZY_FALLBACK)
    target_compile_definitions(lw_ppocr_c PRIVATE LW_WASM_REC_LAZY_FALLBACK=1)
endif()
lw_configure_runtime_features(lw_ppocr_c)

if(NOT EMSCRIPTEN AND NOT LW_RUNTIME_ONLY)
add_library(lw_ppocr_c_shared SHARED ${LW_PPOCR_C_SOURCES})
set_target_properties(lw_ppocr_c_shared PROPERTIES
    EXPORT_NAME shared
    OUTPUT_NAME lw_ppocr_c
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    C_VISIBILITY_PRESET hidden)
target_compile_definitions(lw_ppocr_c_shared
    PRIVATE LW_PPOCR_C_BUILD LW_PPOCR_C_SHARED
    INTERFACE LW_PPOCR_C_SHARED)
lw_configure_runtime_target(lw_ppocr_c_shared)
if(LW_X64_REC_FAST_ENABLED)
    target_compile_definitions(lw_ppocr_c_shared PRIVATE LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
    if(MSVC)
        set_property(SOURCE src/simd/avx2_nhwc_pointwise.c src/simd/avx2_nhwc_dense.c src/simd/avx2_nhwc_depthwise.c src/simd/avx2_nhwc_support.c src/simd/avx2_nhwc_convtranspose.c APPEND PROPERTY
            COMPILE_OPTIONS /arch:AVX2 /fp:strict)
    endif()
endif()
lw_configure_runtime_features(lw_ppocr_c_shared)
endif()
