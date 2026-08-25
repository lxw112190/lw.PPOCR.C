cmake_minimum_required(VERSION 3.20)

project(lw.PPOCR.C VERSION 0.1.0 LANGUAGES C)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

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

set(LW_PPOCR_C_SOURCES
    src/kernels/scalar_activation.c
    src/kernels/scalar_conv.c
    src/kernels/scalar_elementwise.c
    src/kernels/scalar_layout.c
    src/kernels/scalar_matmul.c
    src/kernels/scalar_reduction.c
    src/kernels/scalar_softmax.c
    src/ppocr/ctc_decode.c
    src/ppocr/rec_preprocess.c
    src/ppocr/recognizer.c
    src/runtime/executor.c
    src/runtime/memory.c
    src/runtime/model.c
    src/runtime/session.c
    src/runtime/shape.c
    src/runtime/validate.c)

function(lw_configure_runtime_target target)
    target_include_directories(${target}
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(${target} PUBLIC c_std_11)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_link_libraries(${target} PUBLIC m)
    endif()
endfunction()

add_library(lw_ppocr_c STATIC ${LW_PPOCR_C_SOURCES})
set_target_properties(lw_ppocr_c PROPERTIES
    EXPORT_NAME static
    OUTPUT_NAME lw_ppocr_c_static)
lw_configure_runtime_target(lw_ppocr_c)

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
target_link_libraries(lw-rec-benchmark PRIVATE lw_ppocr_c_shared)
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
if(MSVC)
    target_compile_options(lw-rec-benchmark PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-rec-benchmark PRIVATE
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

if(BUILD_TESTING)
    add_test(
        NAME model_analysis
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_analyze_onnx)
    set_tests_properties(model_analysis PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME rec_converter
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_convert_rec)
    set_tests_properties(rec_converter PROPERTIES
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
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
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
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
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
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
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
endif()

if(MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)
    include(InstallRequiredSystemLibraries)
endif()

install(TARGETS lw_ppocr_c lw_ppocr_c_shared
    EXPORT lw.PPOCR.CTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(TARGETS lw-recognize-ppm lw-rec-benchmark
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES include/lw_infer.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES
    examples/CMakeLists.txt
    examples/benchmark_ppm.c
    examples/ppm_image.h
    examples/recognize_ppm.c
    DESTINATION examples)
install(FILES
    "${LW_PPOCR_C_REC_LWM}"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
    "${LW_PPOCR_C_REC_DEMO_PPM}"
    DESTINATION models)
install(DIRECTORY docs/
    DESTINATION docs
    FILES_MATCHING PATTERN "*.md")
install(FILES
    README.md
    LICENSE
    THIRD-PARTY-NOTICES.md
    dependencies.lock.json
    sbom.cdx.json
    DESTINATION .)
install(FILES licenses/PaddleOCR-models-APACHE-2.0.txt
    DESTINATION licenses)

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

if(BUILD_TESTING)
    add_test(
        NAME staged_package
        COMMAND "${CMAKE_COMMAND}"
                -DLW_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
                -DLW_STAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}/stage-test
                -DLW_CONFIG=$<CONFIG>
                -DLW_PYTHON=${Python3_EXECUTABLE}
                -DLW_TEST_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tests/test_staged_package.py
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
