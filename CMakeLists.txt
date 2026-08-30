cmake_minimum_required(VERSION 3.20)

project(lw.PPOCR.C VERSION 0.1.0 LANGUAGES C)

option(LW_BUILD_HTTP_DEMO
    "Build the cross-platform cpp-httplib HTTP OCR Demo" ON)
option(LW_BUILD_CSHARP_DEMOS
    "Build the .NET Framework 3.5 WinForms Demo on Windows" OFF)
option(LW_WASM_SIMD128
    "Enable portable WebAssembly SIMD128 for the standalone OCR HTML" ON)

if(LW_BUILD_HTTP_DEMO)
    enable_language(CXX)
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
if(NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
endif()

set(LW_PPOCR_C_ANALYSIS_JSON
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/ppocrv6-tiny-analysis.json")
set(LW_PPOCR_C_ANALYSIS_MARKDOWN
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/SUPPORTED_OPS_V0.md")
set(LW_PPOCR_C_REC_ONNX
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/rec.onnx")
set(LW_PPOCR_C_REC_LWM
    "${CMAKE_CURRENT_BINARY_DIR}/models/rec.lwm")
set(LW_PPOCR_C_REC_LWM_METADATA
    "${CMAKE_CURRENT_BINARY_DIR}/models/rec-lwm.json")
set(LW_PPOCR_C_CLS_ONNX
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/cls.onnx")
set(LW_PPOCR_C_CLS_LWM
    "${CMAKE_CURRENT_BINARY_DIR}/models/cls.lwm")
set(LW_PPOCR_C_CLS_LWM_METADATA
    "${CMAKE_CURRENT_BINARY_DIR}/models/cls-lwm.json")
set(LW_PPOCR_C_DET_ONNX
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/det.onnx")
set(LW_PPOCR_C_DET_LWM
    "${CMAKE_CURRENT_BINARY_DIR}/models/det.lwm")
set(LW_PPOCR_C_DET_LWM_METADATA
    "${CMAKE_CURRENT_BINARY_DIR}/models/det-lwm.json")

add_custom_target(analyze_models
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/converter/analyze_onnx.py"
            --json-output "${LW_PPOCR_C_ANALYSIS_JSON}"
            --markdown-output "${LW_PPOCR_C_ANALYSIS_MARKDOWN}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Analyzing bundled PP-OCRv6 tiny ONNX models"
    VERBATIM)

add_custom_command(
    OUTPUT "${LW_PPOCR_C_REC_LWM}" "${LW_PPOCR_C_REC_LWM_METADATA}"
    COMMAND "${Python3_EXECUTABLE}" -m converter.convert_rec
            --input "${LW_PPOCR_C_REC_ONNX}"
            --output "${LW_PPOCR_C_REC_LWM}"
            --metadata-output "${LW_PPOCR_C_REC_LWM_METADATA}"
    DEPENDS
        "${LW_PPOCR_C_REC_ONNX}"
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/convert_rec.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/lwm_v0.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Converting PP-OCRv6 tiny REC to LWM v0.1"
    VERBATIM)
add_custom_target(convert_rec_model ALL
    DEPENDS "${LW_PPOCR_C_REC_LWM}" "${LW_PPOCR_C_REC_LWM_METADATA}")

add_custom_command(
    OUTPUT "${LW_PPOCR_C_CLS_LWM}" "${LW_PPOCR_C_CLS_LWM_METADATA}"
    COMMAND "${Python3_EXECUTABLE}" -m converter.convert_cls
            --input "${LW_PPOCR_C_CLS_ONNX}"
            --output "${LW_PPOCR_C_CLS_LWM}"
            --metadata-output "${LW_PPOCR_C_CLS_LWM_METADATA}"
    DEPENDS
        "${LW_PPOCR_C_CLS_ONNX}"
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/convert_cls.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/lwm_v0.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Converting PP-OCRv6 tiny CLS to LWM v0.1"
    VERBATIM)
add_custom_target(convert_cls_model ALL
    DEPENDS "${LW_PPOCR_C_CLS_LWM}" "${LW_PPOCR_C_CLS_LWM_METADATA}")

add_custom_command(
    OUTPUT "${LW_PPOCR_C_DET_LWM}" "${LW_PPOCR_C_DET_LWM_METADATA}"
    COMMAND "${Python3_EXECUTABLE}" -m converter.convert_det
            --input "${LW_PPOCR_C_DET_ONNX}"
            --output "${LW_PPOCR_C_DET_LWM}"
            --metadata-output "${LW_PPOCR_C_DET_LWM_METADATA}"
    DEPENDS
        "${LW_PPOCR_C_DET_ONNX}"
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/convert_det.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/lwm_v0.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Converting PP-OCRv6 tiny DET to LWM v0.1"
    VERBATIM)
add_custom_target(convert_det_model ALL
    DEPENDS "${LW_PPOCR_C_DET_LWM}" "${LW_PPOCR_C_DET_LWM_METADATA}")

set(LW_PPOCR_C_SOURCES
    src/kernels/packed_conv.c
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
    src/ppocr/db_postprocess.c
    src/ppocr/det_preprocess.c
    src/ppocr/detector.c
    src/ppocr/ocr.c
    src/ppocr/profile.c
    src/ppocr/rec_preprocess.c
    src/ppocr/recognizer.c
    src/runtime/executor.c
    src/runtime/memory.c
    src/runtime/model.c
    src/runtime/parallel.c
    src/runtime/session.c
    src/runtime/shape.c
    src/runtime/validate.c
    src/simd/avx2_binary.c
    src/simd/avx2_conv1x1.c
    src/simd/avx2_conv2x2.c
    src/simd/avx2_conv3x3.c
    src/simd/avx2_convtranspose2x2.c
    src/simd/avx2_depthwise3x3.c
    src/simd/avx2_depthwise5x5.c
    src/simd/avx2_erf.c
    src/simd/avx2_softmax.c
    src/simd/avx2_matmul.c
    src/simd/avx2_packed_conv1x1.c
    src/simd/cpu_features.c
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

if(MSVC AND MSVC_C_ARCHITECTURE_ID MATCHES "^(x64|X86)$")
    set_property(SOURCE src/simd/avx2_binary.c src/simd/avx2_conv1x1.c
        src/simd/avx2_conv2x2.c
        src/simd/avx2_conv3x3.c src/simd/avx2_convtranspose2x2.c
        src/simd/avx2_depthwise3x3.c
        src/simd/avx2_depthwise5x5.c
        src/simd/avx2_erf.c src/simd/avx2_matmul.c
        src/simd/avx2_packed_conv1x1.c APPEND PROPERTY
        COMPILE_OPTIONS /arch:AVX2 /fp:strict)
    set_property(SOURCE src/simd/cpu_features.c APPEND PROPERTY
        COMPILE_OPTIONS /arch:AVX)
endif()

function(lw_configure_runtime_target target)
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

add_library(lw_ppocr_c STATIC ${LW_PPOCR_C_SOURCES})
set_target_properties(lw_ppocr_c PROPERTIES
    EXPORT_NAME static
    OUTPUT_NAME lw_ppocr_c_static)
lw_configure_runtime_target(lw_ppocr_c)

if(NOT EMSCRIPTEN)
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
endif()

if(EMSCRIPTEN)
    add_executable(lw_ppocr_web web/lw_web_api.c)
    target_link_libraries(lw_ppocr_web PRIVATE lw_ppocr_c)
    target_include_directories(lw_ppocr_web PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(lw_ppocr_web PRIVATE c_std_11)
    target_compile_options(lw_ppocr_web PRIVATE -O3 -Wall -Wextra -Wpedantic -Werror)
    if(LW_WASM_SIMD128)
        target_compile_options(lw_ppocr_web PRIVATE -msimd128 -msse2)
        target_link_options(lw_ppocr_web PRIVATE -msimd128)
    endif()
    target_link_options(lw_ppocr_web PRIVATE
        -O3
        "-sMODULARIZE=1" "-sEXPORT_NAME=LwPpocrModule" "-sSINGLE_FILE=1"
        "-sFORCE_FILESYSTEM=1" "-sALLOW_MEMORY_GROWTH=1" "-sNO_EXIT_RUNTIME=1"
        "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8','HEAPU32','HEAPF32']"
        "-sEXPORTED_FUNCTIONS=['_lw_web_init','_lw_web_shutdown','_lw_web_info_size','_lw_web_get_info','_lw_web_run','_lw_web_last_error','_lw_web_malloc','_lw_web_free']"
        "--no-entry")
    set_target_properties(lw_ppocr_web PROPERTIES OUTPUT_NAME "lw_ppocr_web")
    set_property(TARGET lw_ppocr_web APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo.template.html"
        "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_html.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/sponsor.jpg")
    add_dependencies(lw_ppocr_web convert_det_model convert_cls_model convert_rec_model)
    add_custom_command(TARGET lw_ppocr_web POST_BUILD
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_html.py"
            --template "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo.template.html"
            --runtime "$<TARGET_FILE_DIR:lw_ppocr_web>/lw_ppocr_web.js"
            --det "${LW_PPOCR_C_DET_LWM}" --cls "${LW_PPOCR_C_CLS_LWM}"
            --rec "${LW_PPOCR_C_REC_LWM}"
            --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
            --sponsor "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/sponsor.jpg"
            --output "$<TARGET_FILE_DIR:lw_ppocr_web>/ocr-demo.html"
        VERBATIM)
    add_custom_target(lw-ocr-html DEPENDS lw_ppocr_web)
endif()

set(LW_PPOCR_C_REC_DEMO_PPM
    "${CMAKE_CURRENT_BINARY_DIR}/models/sample-crop.ppm")
add_custom_command(
    OUTPUT "${LW_PPOCR_C_REC_DEMO_PPM}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_rec_demo_fixture.py"
            --input "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
            --output "${LW_PPOCR_C_REC_DEMO_PPM}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_rec_demo_fixture.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
    COMMENT "Creating dependency-free REC demo PPM fixture"
    VERBATIM)
add_custom_target(rec_demo_fixture ALL DEPENDS "${LW_PPOCR_C_REC_DEMO_PPM}")

set(LW_PPOCR_C_DET_DEMO_PPM
    "${CMAKE_CURRENT_BINARY_DIR}/models/sample.ppm")
add_custom_command(
    OUTPUT "${LW_PPOCR_C_DET_DEMO_PPM}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_det_demo_fixture.py"
            --input "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
            --output "${LW_PPOCR_C_DET_DEMO_PPM}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_det_demo_fixture.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
    COMMENT "Creating dependency-free DET demo PPM fixture"
    VERBATIM)
add_custom_target(det_demo_fixture ALL DEPENDS "${LW_PPOCR_C_DET_DEMO_PPM}")

if(NOT EMSCRIPTEN)
add_executable(lw-recognize-ppm examples/recognize_ppm.c)
target_link_libraries(lw-recognize-ppm PRIVATE lw_ppocr_c_shared)
target_compile_features(lw-recognize-ppm PRIVATE c_std_11)
add_dependencies(lw-recognize-ppm convert_rec_model rec_demo_fixture)
if(WIN32)
    target_link_libraries(lw-recognize-ppm PRIVATE shell32)
elseif(APPLE)
    set_target_properties(lw-recognize-ppm PROPERTIES INSTALL_RPATH "@loader_path/../lib")
else()
    set_target_properties(lw-recognize-ppm PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

add_executable(lw-rec-benchmark examples/benchmark_ppm.c)
target_sources(lw-rec-benchmark PRIVATE src/simd/cpu_features.c)
target_link_libraries(lw-rec-benchmark PRIVATE lw_ppocr_c_shared)
target_include_directories(lw-rec-benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
target_compile_features(lw-rec-benchmark PRIVATE c_std_11)
add_dependencies(lw-rec-benchmark convert_rec_model rec_demo_fixture)
if(WIN32)
    target_link_libraries(lw-rec-benchmark PRIVATE shell32 psapi)
elseif(APPLE)
    set_target_properties(lw-rec-benchmark PROPERTIES
        INSTALL_RPATH "@loader_path/../lib")
else()
    set_target_properties(lw-rec-benchmark PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib")
endif()

add_executable(lw-ocr-benchmark examples/ocr_benchmark_ppm.c)
target_sources(lw-ocr-benchmark PRIVATE src/simd/cpu_features.c)
target_link_libraries(lw-ocr-benchmark PRIVATE lw_ppocr_c_shared)
target_include_directories(lw-ocr-benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
target_compile_features(lw-ocr-benchmark PRIVATE c_std_11)
add_dependencies(lw-ocr-benchmark convert_det_model convert_cls_model
    convert_rec_model det_demo_fixture)
if(WIN32)
    target_link_libraries(lw-ocr-benchmark PRIVATE shell32 psapi)
elseif(APPLE)
    set_target_properties(lw-ocr-benchmark PROPERTIES
        INSTALL_RPATH "@loader_path/../lib")
else()
    set_target_properties(lw-ocr-benchmark PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib")
endif()

add_executable(lw-classify-ppm examples/classify_ppm.c)
target_link_libraries(lw-classify-ppm PRIVATE lw_ppocr_c_shared)
target_compile_features(lw-classify-ppm PRIVATE c_std_11)
add_dependencies(lw-classify-ppm convert_cls_model rec_demo_fixture)
if(WIN32)
    target_link_libraries(lw-classify-ppm PRIVATE shell32)
elseif(APPLE)
    set_target_properties(lw-classify-ppm PROPERTIES INSTALL_RPATH "@loader_path/../lib")
else()
    set_target_properties(lw-classify-ppm PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

add_executable(lw-detect-ppm examples/detect_ppm.c)
target_link_libraries(lw-detect-ppm PRIVATE lw_ppocr_c_shared)
target_compile_features(lw-detect-ppm PRIVATE c_std_11)
add_dependencies(lw-detect-ppm convert_det_model det_demo_fixture)
if(WIN32)
    target_link_libraries(lw-detect-ppm PRIVATE shell32)
elseif(APPLE)
    set_target_properties(lw-detect-ppm PROPERTIES INSTALL_RPATH "@loader_path/../lib")
else()
    set_target_properties(lw-detect-ppm PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

add_executable(lw-ocr-ppm examples/ocr_ppm.c)
target_link_libraries(lw-ocr-ppm PRIVATE lw_ppocr_c_shared)
target_compile_features(lw-ocr-ppm PRIVATE c_std_11)
add_dependencies(lw-ocr-ppm convert_det_model convert_cls_model
    convert_rec_model det_demo_fixture)
if(WIN32)
    target_link_libraries(lw-ocr-ppm PRIVATE shell32)
elseif(APPLE)
    set_target_properties(lw-ocr-ppm PROPERTIES INSTALL_RPATH "@loader_path/../lib")
else()
    set_target_properties(lw-ocr-ppm PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()
if(MSVC)
    target_compile_options(lw-classify-ppm PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-classify-ppm PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
if(MSVC)
    target_compile_options(lw-detect-ppm PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-detect-ppm PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
if(MSVC)
    target_compile_options(lw-ocr-ppm PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-ocr-ppm PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
if(MSVC)
    target_compile_options(lw-rec-benchmark PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-rec-benchmark PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
if(MSVC)
    target_compile_options(lw-ocr-benchmark PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-ocr-benchmark PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
if(MSVC)
    target_compile_options(lw-recognize-ppm PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-recognize-ppm PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(lwm-inspect tools/lwm_inspect.c)
target_link_libraries(lwm-inspect PRIVATE lw_ppocr_c)
if(WIN32)
    target_link_libraries(lwm-inspect PRIVATE shell32)
endif()
target_compile_features(lwm-inspect PRIVATE c_std_11)
if(MSVC)
    target_compile_options(lwm-inspect PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lwm-inspect PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(LW_BUILD_HTTP_DEMO)
    set(LW_HTTPLIB_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/http-server/third_party/httplib.h")
    file(SHA256 "${LW_HTTPLIB_HEADER}" LW_HTTPLIB_SHA256)
    if(NOT LW_HTTPLIB_SHA256 STREQUAL
       "1c8b44c8ab0498322c08840dbdfcd5006bc28c5a3b140b0c2cf30777b7232fc6")
        message(FATAL_ERROR "vendored cpp-httplib checksum mismatch")
    endif()
    configure_file(
        examples/http-server/www/index.html
        "${CMAKE_CURRENT_BINARY_DIR}/www/index.html"
        COPYONLY)
    add_executable(lw-http-ocr examples/http-server/main.cpp)
    set_target_properties(lw-http-ocr PROPERTIES
        OUTPUT_NAME "lw.PPOCR.C.HttpServer"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
    foreach(lw_config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${lw_config}" lw_config_upper)
        set_target_properties(lw-http-ocr PROPERTIES
            "RUNTIME_OUTPUT_DIRECTORY_${lw_config_upper}"
            "${CMAKE_CURRENT_BINARY_DIR}/bin")
    endforeach()
    target_include_directories(lw-http-ocr PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_include_directories(lw-http-ocr SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/http-server/third_party")
    target_link_libraries(lw-http-ocr PRIVATE lw_ppocr_c)
    target_compile_features(lw-http-ocr PRIVATE cxx_std_11)
    add_dependencies(lw-http-ocr convert_det_model convert_cls_model
        convert_rec_model)
    if(WIN32)
        target_compile_definitions(lw-http-ocr PRIVATE
            WINVER=0x0601 _WIN32_WINNT=0x0601
            WIN32_LEAN_AND_MEAN NOMINMAX)
        target_link_libraries(lw-http-ocr PRIVATE ws2_32)
    elseif(UNIX AND NOT APPLE)
        target_link_libraries(lw-http-ocr PRIVATE pthread)
    endif()
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        target_compile_definitions(lw-http-ocr PRIVATE
            LW_HTTP_THREAD_POOL_COUNT=2)
    endif()
    if(MSVC)
        target_compile_options(lw-http-ocr PRIVATE /W4 /WX /utf-8)
        target_link_options(lw-http-ocr PRIVATE
            /SUBSYSTEM:CONSOLE,6.01
            $<$<EQUAL:${CMAKE_SIZEOF_VOID_P},4>:/LARGEADDRESSAWARE>)
    else()
        target_compile_options(lw-http-ocr PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_custom_command(TARGET lw-http-ocr POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                "${CMAKE_CURRENT_BINARY_DIR}/models/ppocr_keys.txt"
        VERBATIM)
endif()

if(LW_BUILD_CSHARP_DEMOS)
    if(NOT WIN32)
        message(FATAL_ERROR "LW_BUILD_CSHARP_DEMOS is supported only on Windows")
    endif()
    find_program(LW_MSBUILD_EXECUTABLE NAMES MSBuild.exe msbuild REQUIRED)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(LW_MANAGED_PLATFORM x64)
    else()
        set(LW_MANAGED_PLATFORM x86)
    endif()
    set(LW_MANAGED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/managed")
    set(LW_MANAGED_WINFORMS_DIR "${LW_MANAGED_ROOT}/winforms")
    set(LW_MANAGED_MODELS_DIR "${LW_MANAGED_ROOT}/models")
    set(LW_MANAGED_WINFORMS_EXE
        "${LW_MANAGED_WINFORMS_DIR}/lw.PPOCR.C.WinForms.exe")
    add_custom_target(lw-csharp-demos ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${LW_MANAGED_WINFORMS_DIR}" "${LW_MANAGED_MODELS_DIR}"
        COMMAND "${LW_MSBUILD_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/csharp-winforms/LwPpocrWinForms.csproj"
                -t:Restore -p:RestorePackagesConfig=true
                -p:RestoreIgnoreFailedSources=true -p:NuGetAudit=false -v:minimal
        COMMAND "${LW_MSBUILD_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/csharp-winforms/LwPpocrWinForms.csproj"
                -t:Rebuild -p:Configuration=Release
                -p:Platform=${LW_MANAGED_PLATFORM}
                "-p:OutputPath=${LW_MANAGED_WINFORMS_DIR}/" -v:minimal
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:lw_ppocr_c_shared>"
                "${LW_MANAGED_WINFORMS_DIR}/"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${LW_PPOCR_C_DET_LWM}" "${LW_PPOCR_C_CLS_LWM}"
                "${LW_PPOCR_C_REC_LWM}"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
                "${LW_MANAGED_MODELS_DIR}/"
        DEPENDS
            lw_ppocr_c_shared
            convert_det_model convert_cls_model convert_rec_model
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/csharp-common/NativeOcr.cs"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/csharp-common/OcrModels.cs"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/csharp-winforms/MainForm.cs"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/csharp-winforms/OcrResultExporter.cs"
        COMMENT "Building .NET Framework 3.5 WinForms OCR Demo"
        VERBATIM)
endif()

endif()

if(BUILD_TESTING AND NOT EMSCRIPTEN)
    add_test(
        NAME model_analysis
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_analyze_onnx)
    set_tests_properties(model_analysis PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME conv_shape_analysis
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_analyze_conv_shapes)
    set_tests_properties(conv_shape_analysis PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME conv_bn_fusion
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_conv_bn_fusion)
    set_tests_properties(conv_bn_fusion PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME rec_converter
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_convert_rec)
    set_tests_properties(rec_converter PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME cls_converter
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_convert_cls)
    set_tests_properties(cls_converter PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME det_converter
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_convert_det)
    set_tests_properties(det_converter PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(test-lwm-loader tests/test_model.c)
    target_link_libraries(test-lwm-loader PRIVATE lw_ppocr_c)
    target_compile_features(test-lwm-loader PRIVATE c_std_11)
    add_dependencies(test-lwm-loader convert_rec_model)
    if(MSVC)
        target_compile_options(test-lwm-loader PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-lwm-loader PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME lwm_loader COMMAND test-lwm-loader "${LW_PPOCR_C_REC_LWM}")

    add_executable(test-session-planner tests/test_session.c)
    target_link_libraries(test-session-planner PRIVATE lw_ppocr_c)
    target_compile_features(test-session-planner PRIVATE c_std_11)
    add_dependencies(test-session-planner convert_rec_model)
    if(MSVC)
        target_compile_options(test-session-planner PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-session-planner PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME session_planner COMMAND test-session-planner "${LW_PPOCR_C_REC_LWM}")

    add_executable(test-c-abi tests/test_abi.c)
    target_link_libraries(test-c-abi PRIVATE lw_ppocr_c)
    target_compile_features(test-c-abi PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(test-c-abi PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-c-abi PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME c_abi_contract COMMAND test-c-abi)

    add_executable(test-recognizer-api tests/test_recognizer.c)
    target_link_libraries(test-recognizer-api PRIVATE lw_ppocr_c)
    target_compile_features(test-recognizer-api PRIVATE c_std_11)
    add_dependencies(test-recognizer-api convert_rec_model)
    if(MSVC)
        target_compile_options(test-recognizer-api PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-recognizer-api PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME recognizer_api_contract
        COMMAND test-recognizer-api
                "${LW_PPOCR_C_REC_LWM}"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt")

    add_executable(test-classifier-api tests/test_classifier.c)
    target_link_libraries(test-classifier-api PRIVATE lw_ppocr_c)
    target_compile_features(test-classifier-api PRIVATE c_std_11)
    add_dependencies(test-classifier-api convert_cls_model)
    if(MSVC)
        target_compile_options(test-classifier-api PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-classifier-api PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME classifier_api_contract
        COMMAND test-classifier-api "${LW_PPOCR_C_CLS_LWM}")

    add_test(
        NAME c_demo_classification
        COMMAND lw-classify-ppm
                "${LW_PPOCR_C_CLS_LWM}"
                "${LW_PPOCR_C_REC_DEMO_PPM}")

    add_test(
        NAME c_demo_detection
        COMMAND lw-detect-ppm
                "${LW_PPOCR_C_DET_LWM}"
                "${LW_PPOCR_C_DET_DEMO_PPM}")
    set_tests_properties(c_demo_detection PROPERTIES TIMEOUT 300)

    add_test(
        NAME c_demo_full_ocr
        COMMAND lw-ocr-ppm
                "${LW_PPOCR_C_DET_LWM}"
                "${LW_PPOCR_C_CLS_LWM}"
                "${LW_PPOCR_C_REC_LWM}"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                "${LW_PPOCR_C_DET_DEMO_PPM}")
    set_tests_properties(c_demo_full_ocr PROPERTIES TIMEOUT 600)

    add_test(
        NAME c_demo_recognition
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_recognize_ppm_demo.py"
                --demo "$<TARGET_FILE:lw-recognize-ppm>"
                --model "${LW_PPOCR_C_REC_LWM}"
                --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --image "${LW_PPOCR_C_REC_DEMO_PPM}")
    set_tests_properties(c_demo_recognition PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 180)

    add_test(
        NAME rec_scalar_benchmark
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rec_benchmark.py"
                --benchmark "$<TARGET_FILE:lw-rec-benchmark>"
                --model "${LW_PPOCR_C_REC_LWM}"
                --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --image "${LW_PPOCR_C_REC_DEMO_PPM}")
    set_tests_properties(rec_scalar_benchmark PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 240)

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(LW_FULL_OCR_BENCH_WORKERS 8)
    else()
        set(LW_FULL_OCR_BENCH_WORKERS 1)
    endif()
    add_test(
        NAME full_ocr_parallel_benchmark
        COMMAND lw-ocr-benchmark
                "${LW_PPOCR_C_DET_LWM}"
                "${LW_PPOCR_C_CLS_LWM}"
                "${LW_PPOCR_C_REC_LWM}"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                "${LW_PPOCR_C_DET_DEMO_PPM}"
                1 1 "${LW_FULL_OCR_BENCH_WORKERS}" 960)
    set_tests_properties(full_ocr_parallel_benchmark PROPERTIES TIMEOUT 300)

    if(MSVC)
        get_filename_component(LW_EXPORT_TOOL_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
        set(LW_EXPORT_TOOL "${LW_EXPORT_TOOL_DIR}/dumpbin.exe")
    else()
        set(LW_EXPORT_TOOL "${CMAKE_NM}")
    endif()
    add_test(
        NAME shared_export_contract
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shared_exports.py"
                --library "$<TARGET_FILE:lw_ppocr_c_shared>"
                --allowlist "${CMAKE_CURRENT_SOURCE_DIR}/abi/exports-v0.txt"
                --tool "${LW_EXPORT_TOOL}")
    set_tests_properties(shared_export_contract PROPERTIES TIMEOUT 60)

    add_executable(kernel-reference-driver tests/kernel_reference_driver.c)
    target_link_libraries(kernel-reference-driver PRIVATE lw_ppocr_c)
    target_include_directories(kernel-reference-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(kernel-reference-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(kernel-reference-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(kernel-reference-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME scalar_kernel_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scalar_kernels_reference.py"
                --driver "$<TARGET_FILE:kernel-reference-driver>")
    set_tests_properties(scalar_kernel_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(tensor-reference-driver tests/tensor_reference_driver.c)
    target_link_libraries(tensor-reference-driver PRIVATE lw_ppocr_c)
    target_include_directories(tensor-reference-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(tensor-reference-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(tensor-reference-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(tensor-reference-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME tensor_kernel_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor_kernels_reference.py"
                --driver "$<TARGET_FILE:tensor-reference-driver>")
    set_tests_properties(tensor_kernel_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(conv-reference-driver tests/conv_reference_driver.c)
    target_link_libraries(conv-reference-driver PRIVATE lw_ppocr_c)
    target_include_directories(conv-reference-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(conv-reference-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(conv-reference-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(conv-reference-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME conv_kernel_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv_kernels_reference.py"
                --driver "$<TARGET_FILE:conv-reference-driver>")
    set_tests_properties(conv_kernel_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(rec-graph-driver tests/rec_graph_driver.c)
    target_link_libraries(rec-graph-driver PRIVATE lw_ppocr_c)
    target_include_directories(rec-graph-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(rec-graph-driver PRIVATE c_std_11)
    add_dependencies(rec-graph-driver convert_rec_model)
    if(MSVC)
        target_compile_options(rec-graph-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(rec-graph-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME rec_graph_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rec_graph_reference.py"
                --driver "$<TARGET_FILE:rec-graph-driver>"
                --lwm-model "${LW_PPOCR_C_REC_LWM}"
                --onnx-model "${LW_PPOCR_C_REC_ONNX}")
    set_tests_properties(rec_graph_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 360)

    add_executable(cls-graph-driver tests/cls_graph_driver.c)
    target_link_libraries(cls-graph-driver PRIVATE lw_ppocr_c)
    target_include_directories(cls-graph-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(cls-graph-driver PRIVATE c_std_11)
    add_dependencies(cls-graph-driver convert_cls_model)
    if(MSVC)
        target_compile_options(cls-graph-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(cls-graph-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME cls_graph_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cls_graph_reference.py"
                --driver "$<TARGET_FILE:cls-graph-driver>"
                --lwm-model "${LW_PPOCR_C_CLS_LWM}"
                --onnx-model "${LW_PPOCR_C_CLS_ONNX}")
    set_tests_properties(cls_graph_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 360)

    add_executable(det-graph-driver tests/det_graph_driver.c)
    target_link_libraries(det-graph-driver PRIVATE lw_ppocr_c)
    target_include_directories(det-graph-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(det-graph-driver PRIVATE c_std_11)
    add_dependencies(det-graph-driver convert_det_model)
    if(MSVC)
        target_compile_options(det-graph-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(det-graph-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME det_graph_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_det_graph_reference.py"
                --driver "$<TARGET_FILE:det-graph-driver>"
                --lwm-model "${LW_PPOCR_C_DET_LWM}"
                --onnx-model "${LW_PPOCR_C_DET_ONNX}")
    set_tests_properties(det_graph_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 600)

    add_executable(det-pipeline-driver tests/det_pipeline_driver.c)
    target_link_libraries(det-pipeline-driver PRIVATE lw_ppocr_c)
    target_include_directories(det-pipeline-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(det-pipeline-driver PRIVATE c_std_11)
    add_dependencies(det-pipeline-driver convert_det_model)
    if(MSVC)
        target_compile_options(det-pipeline-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(det-pipeline-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME det_pipeline_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_det_pipeline_reference.py"
                --driver "$<TARGET_FILE:det-pipeline-driver>"
                --model "${LW_PPOCR_C_DET_LWM}"
                --sample "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg")
    set_tests_properties(det_pipeline_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 900)

    add_executable(ocr-pipeline-driver tests/ocr_pipeline_driver.c)
    target_link_libraries(ocr-pipeline-driver PRIVATE lw_ppocr_c)
    target_include_directories(ocr-pipeline-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(ocr-pipeline-driver PRIVATE c_std_11)
    add_dependencies(ocr-pipeline-driver convert_det_model convert_cls_model
        convert_rec_model)
    if(MSVC)
        target_compile_options(ocr-pipeline-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(ocr-pipeline-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME full_ocr_pipeline_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ocr_pipeline_reference.py"
                --driver "$<TARGET_FILE:ocr-pipeline-driver>"
                --detector "${LW_PPOCR_C_DET_LWM}"
                --classifier "${LW_PPOCR_C_CLS_LWM}"
                --recognizer "${LW_PPOCR_C_REC_LWM}"
                --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --sample "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg")
    set_tests_properties(full_ocr_pipeline_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 1200)

    add_test(
        NAME full_ocr_golden_corpus
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ocr_golden_corpus.py"
                --driver "$<TARGET_FILE:ocr-pipeline-driver>"
                --detector "${LW_PPOCR_C_DET_LWM}"
                --classifier "${LW_PPOCR_C_CLS_LWM}"
                --recognizer "${LW_PPOCR_C_REC_LWM}"
                --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --detector-source "${LW_PPOCR_C_DET_ONNX}"
                --classifier-source "${LW_PPOCR_C_CLS_ONNX}"
                --recognizer-source "${LW_PPOCR_C_REC_ONNX}"
                --sample "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
                --corpus "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ocr-golden-corpus.json")
    set_tests_properties(full_ocr_golden_corpus PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 1200)

    add_executable(cls-pipeline-driver tests/cls_pipeline_driver.c)
    target_link_libraries(cls-pipeline-driver PRIVATE lw_ppocr_c)
    if(WIN32)
        target_link_libraries(cls-pipeline-driver PRIVATE shell32)
    endif()
    target_include_directories(cls-pipeline-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(cls-pipeline-driver PRIVATE c_std_11)
    add_dependencies(cls-pipeline-driver convert_cls_model)
    if(MSVC)
        target_compile_options(cls-pipeline-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(cls-pipeline-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME cls_pipeline_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cls_pipeline_reference.py"
                --driver "$<TARGET_FILE:cls-pipeline-driver>"
                --lwm-model "${LW_PPOCR_C_CLS_LWM}"
                --onnx-model "${LW_PPOCR_C_CLS_ONNX}")
    set_tests_properties(cls_pipeline_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 360)

    add_executable(rec-profile-driver tests/rec_profile_driver.c)
    target_link_libraries(rec-profile-driver PRIVATE lw_ppocr_c)
    target_include_directories(rec-profile-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(rec-profile-driver PRIVATE c_std_11)
    add_dependencies(rec-profile-driver convert_rec_model)
    if(MSVC)
        target_compile_options(rec-profile-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(rec-profile-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME rec_operator_profile
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rec_profile.py"
                --driver "$<TARGET_FILE:rec-profile-driver>"
                --model "${LW_PPOCR_C_REC_LWM}")
    set_tests_properties(rec_operator_profile PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 240)

    add_executable(full-ocr-profile-driver tests/full_ocr_profile_driver.c)
    target_link_libraries(full-ocr-profile-driver PRIVATE lw_ppocr_c)
    target_include_directories(full-ocr-profile-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(full-ocr-profile-driver PRIVATE c_std_11)
    add_dependencies(full-ocr-profile-driver convert_det_model convert_cls_model
        convert_rec_model det_demo_fixture)
    if(MSVC)
        target_compile_options(full-ocr-profile-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(full-ocr-profile-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME full_ocr_operator_profile
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_full_ocr_profile.py"
                --driver "$<TARGET_FILE:full-ocr-profile-driver>"
                --det-model "${LW_PPOCR_C_DET_LWM}"
                --cls-model "${LW_PPOCR_C_CLS_LWM}"
                --rec-model "${LW_PPOCR_C_REC_LWM}"
                --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --image "${LW_PPOCR_C_DET_DEMO_PPM}")
    set_tests_properties(full_ocr_operator_profile PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 360)

    add_executable(rec-pipeline-driver tests/rec_pipeline_driver.c)
    target_link_libraries(rec-pipeline-driver PRIVATE lw_ppocr_c)
    if(WIN32)
        target_link_libraries(rec-pipeline-driver PRIVATE shell32)
    endif()
    target_include_directories(rec-pipeline-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(rec-pipeline-driver PRIVATE c_std_11)
    add_dependencies(rec-pipeline-driver convert_rec_model)
    if(MSVC)
        target_compile_options(rec-pipeline-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(rec-pipeline-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME rec_pipeline_reference
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rec_pipeline_reference.py"
                --driver "$<TARGET_FILE:rec-pipeline-driver>"
                --lwm-model "${LW_PPOCR_C_REC_LWM}"
                --onnx-model "${LW_PPOCR_C_REC_ONNX}"
                --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --sample "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg")
    set_tests_properties(rec_pipeline_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 360)

    add_test(
        NAME rec_golden_corpus
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rec_golden_corpus.py"
                --driver "$<TARGET_FILE:rec-pipeline-driver>"
                --lwm-model "${LW_PPOCR_C_REC_LWM}"
                --onnx-model "${LW_PPOCR_C_REC_ONNX}"
                --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --sample "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
                --corpus "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/rec-golden-corpus.json")
    set_tests_properties(rec_golden_corpus PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 600)

    add_test(
        NAME lwm_corruption
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lwm_corruption.py"
                --inspect "$<TARGET_FILE:lwm-inspect>"
                --model "${LW_PPOCR_C_REC_LWM}")
    set_tests_properties(lwm_corruption PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    if(LW_BUILD_HTTP_DEMO)
        add_test(
            NAME http_demo_smoke
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http_demo.py"
                    --server "$<TARGET_FILE:lw-http-ocr>"
                    --models "${CMAKE_CURRENT_BINARY_DIR}/models"
                    --www "${CMAKE_CURRENT_BINARY_DIR}/www"
                    --sample "${LW_PPOCR_C_DET_DEMO_PPM}")
        set_tests_properties(http_demo_smoke PROPERTIES TIMEOUT 120)
    endif()
endif()

if(MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)
    include(InstallRequiredSystemLibraries)
endif()

if(NOT EMSCRIPTEN)
install(TARGETS lw_ppocr_c lw_ppocr_c_shared
    EXPORT lw.PPOCR.CTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(TARGETS lw-recognize-ppm lw-rec-benchmark lw-ocr-benchmark lw-classify-ppm lw-detect-ppm
    lw-ocr-ppm
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES include/lw_infer.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES
    examples/CMakeLists.txt
    examples/classify_ppm.c
    examples/detect_ppm.c
    examples/ocr_ppm.c
    examples/ppm_image.h
    examples/recognize_ppm.c
    DESTINATION examples)
install(DIRECTORY
    examples/csharp-common
    examples/csharp-winforms
    examples/http-server
    DESTINATION examples
    PATTERN "bin" EXCLUDE
    PATTERN "obj" EXCLUDE)
install(FILES
    "${LW_PPOCR_C_REC_LWM}"
    "${LW_PPOCR_C_CLS_LWM}"
    "${LW_PPOCR_C_DET_LWM}"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
    "${LW_PPOCR_C_REC_DEMO_PPM}"
    "${LW_PPOCR_C_DET_DEMO_PPM}"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/sample.jpg"
    DESTINATION models)
endif()
if(LW_BUILD_CSHARP_DEMOS)
    install(FILES
        "${LW_MANAGED_WINFORMS_EXE}"
        "${LW_MANAGED_WINFORMS_EXE}.config"
        DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
if(LW_BUILD_HTTP_DEMO)
    install(TARGETS lw-http-ocr
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(DIRECTORY examples/http-server/www/
        DESTINATION www)
endif()
install(DIRECTORY docs/
    DESTINATION docs
    FILES_MATCHING
    PATTERN "*.md"
    PATTERN "*.jpg")
install(FILES
    README.md
    README.zh-CN.md
    LICENSE
    THIRD-PARTY-NOTICES.md
    dependencies.lock.json
    sbom.cdx.json
    DESTINATION .)
install(FILES
    licenses/PaddleOCR-models-APACHE-2.0.txt
    licenses/cpp-httplib-MIT.txt
    DESTINATION licenses)

if(NOT EMSCRIPTEN)
set(LW_CMAKE_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/lw.PPOCR.C")
configure_package_config_file(
    cmake/lw.PPOCR.CConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/lw.PPOCR.CConfig.cmake"
    INSTALL_DESTINATION "${LW_CMAKE_PACKAGE_DIR}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/lw.PPOCR.CConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
install(EXPORT lw.PPOCR.CTargets
    FILE lw.PPOCR.CTargets.cmake
    NAMESPACE lw.PPOCR.C::
    DESTINATION "${LW_CMAKE_PACKAGE_DIR}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lw.PPOCR.CConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lw.PPOCR.CConfigVersion.cmake"
    DESTINATION "${LW_CMAKE_PACKAGE_DIR}")
endif()

if(BUILD_TESTING AND NOT EMSCRIPTEN)
    add_test(
        NAME staged_package
        COMMAND "${CMAKE_COMMAND}"
                -DLW_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
                -DLW_STAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}/stage-test
                -DLW_CONFIG=$<CONFIG>
                -DLW_GENERATOR=${CMAKE_GENERATOR}
                -DLW_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
                -DLW_C_COMPILER=${CMAKE_C_COMPILER}
                -DLW_RC_COMPILER=${CMAKE_RC_COMPILER}
                -DLW_MT=${CMAKE_MT}
                -DLW_PYTHON=${Python3_EXECUTABLE}
                -DLW_TEST_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tests/test_staged_package.py
                -DLW_HTTP_TEST_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http_demo.py
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_staged_package_test.cmake")
    set_tests_properties(staged_package PROPERTIES TIMEOUT 240)
endif()

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(LW_PACKAGE_PLATFORM windows-x64-msvc)
    else()
        set(LW_PACKAGE_PLATFORM windows-x86-msvc)
    endif()
    set(CPACK_GENERATOR ZIP)
else()
    string(TOLOWER "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
        LW_PACKAGE_PLATFORM)
    set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_PACKAGE_NAME "lw.PPOCR.C")
set(CPACK_PACKAGE_VENDOR "天天代码码天天")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME
    "lw.PPOCR.C-${PROJECT_VERSION}-${LW_PACKAGE_PLATFORM}")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/packages")
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
include(CPack)
