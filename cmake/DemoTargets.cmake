# Sample image fixtures, native command-line tools, and application demos.

set(LW_PPOCR_C_REC_DEMO_PPM
    "${CMAKE_CURRENT_BINARY_DIR}/models/sample-crop.ppm")
if(LW_PRECONVERTED_ASSETS_DIR)
    add_custom_command(
        OUTPUT "${LW_PPOCR_C_REC_DEMO_PPM}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/models"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${LW_PRECONVERTED_ASSETS_DIR}/sample-crop.ppm"
                "${LW_PPOCR_C_REC_DEMO_PPM}"
        DEPENDS "${LW_PRECONVERTED_ASSETS_DIR}/sample-crop.ppm"
        COMMENT "Copying preconverted REC demo fixture"
        VERBATIM)
else()
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
endif()
add_custom_target(rec_demo_fixture ALL DEPENDS "${LW_PPOCR_C_REC_DEMO_PPM}")

set(LW_PPOCR_C_DET_DEMO_PPM
    "${CMAKE_CURRENT_BINARY_DIR}/models/sample.ppm")
if(LW_PRECONVERTED_ASSETS_DIR)
    add_custom_command(
        OUTPUT "${LW_PPOCR_C_DET_DEMO_PPM}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/models"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${LW_PRECONVERTED_ASSETS_DIR}/sample.ppm"
                "${LW_PPOCR_C_DET_DEMO_PPM}"
        DEPENDS "${LW_PRECONVERTED_ASSETS_DIR}/sample.ppm"
        COMMENT "Copying preconverted DET demo fixture"
        VERBATIM)
else()
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
endif()
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
                "${LW_PPOCR_C_DET_ONNX}" "${LW_PPOCR_C_CLS_ONNX}" "${LW_PPOCR_C_REC_ONNX}"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/model.json"
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
