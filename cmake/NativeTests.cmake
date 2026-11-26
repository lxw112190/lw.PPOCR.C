# Native contract, reference, smoke, and benchmark test targets.

if(BUILD_TESTING AND NOT EMSCRIPTEN)

    add_executable(wasm-ctc-projection-driver
        tests/test_wasm_ctc_projection.c
        src/simd/wasm128_packed_matmul.c
        src/kernels/packed_matmul.c)
    target_include_directories(wasm-ctc-projection-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
    target_compile_features(wasm-ctc-projection-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(wasm-ctc-projection-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(wasm-ctc-projection-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_link_libraries(wasm-ctc-projection-driver PRIVATE m)
    endif()
    add_test(NAME wasm_ctc_projection COMMAND wasm-ctc-projection-driver)

    add_executable(avx2-ctc-projection-driver tests/test_avx2_ctc_projection.c)
    target_link_libraries(avx2-ctc-projection-driver PRIVATE lw_ppocr_c)
    target_include_directories(avx2-ctc-projection-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(avx2-ctc-projection-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(avx2-ctc-projection-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(avx2-ctc-projection-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_link_libraries(avx2-ctc-projection-driver PRIVATE m)
    endif()
    add_test(NAME avx2_ctc_projection_tails COMMAND avx2-ctc-projection-driver)
    set_tests_properties(avx2_ctc_projection_tails PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(nhwc-depthwise-portable-driver
        tests/test_nhwc_depthwise_portable.c
        src/simd/avx2_nhwc_depthwise.c)
    target_include_directories(nhwc-depthwise-portable-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
    target_compile_features(nhwc-depthwise-portable-driver PRIVATE c_std_11)
    target_compile_definitions(nhwc-depthwise-portable-driver PRIVATE
        LW_NHWC_FORCE_PORTABLE=1)
    if(MSVC)
        target_compile_options(nhwc-depthwise-portable-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(nhwc-depthwise-portable-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME nhwc_depthwise_portable COMMAND nhwc-depthwise-portable-driver)

    foreach(policy IN ITEMS default fine arm64_fine)
        add_executable(rec-width-policy-${policy} tests/rec_width_policy_driver.c)
        target_include_directories(rec-width-policy-${policy} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
        target_compile_features(rec-width-policy-${policy} PRIVATE c_std_11)
        if(policy STREQUAL "fine")
            target_compile_definitions(rec-width-policy-${policy} PRIVATE
                LW_NATIVE_FINE_REC_WIDTHS=1 LW_TEST_FINE_WIDTHS=1)
        elseif(policy STREQUAL "arm64_fine")
            target_compile_definitions(rec-width-policy-${policy} PRIVATE
                LW_ARM64_FINE_REC_WIDTHS=1 LW_TEST_FINE_WIDTHS=1)
        endif()
        if(MSVC)
            target_compile_options(rec-width-policy-${policy} PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(rec-width-policy-${policy} PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME rec_width_policy_${policy} COMMAND rec-width-policy-${policy})
    endforeach()
    add_test(NAME onnx_e2e_comparison_contract
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v tests.test_onnx_e2e_comparison)
    set_tests_properties(onnx_e2e_comparison_contract PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 30)
    if(WIN32 AND MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND
       MSVC_C_ARCHITECTURE_ID MATCHES "^(x64|X86)$")
        add_test(NAME csharp_image_limits_contract
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_csharp_image_limits.py"
                --native "$<TARGET_FILE:lw_ppocr_c_shared>")
        set_tests_properties(csharp_image_limits_contract PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 240)
    endif()
    add_test(NAME web_release_profile_contract
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v tests.test_web_release_profile)
    set_tests_properties(web_release_profile_contract PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 30)
    add_test(NAME winforms_release_version_contract
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v tests.test_winforms_release)
    set_tests_properties(winforms_release_version_contract PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 30)
    add_test(NAME native_rec_quality_contract
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v tests.test_native_rec_quality)
    set_tests_properties(native_rec_quality_contract PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 30)

    add_test(
        NAME source_final_newline_contract
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_source_newlines.py"
                --root "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(source_final_newline_contract PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 30)
    set(LW_RESHAPE_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-fixtures")
    set(LW_RESHAPE_FIXTURES
        "${LW_RESHAPE_FIXTURE_DIR}/valid.lwm"
        "${LW_RESHAPE_FIXTURE_DIR}/two-unknown.lwm"
        "${LW_RESHAPE_FIXTURE_DIR}/non-divisible.lwm"
        "${LW_RESHAPE_FIXTURE_DIR}/zero.lwm"
        "${LW_RESHAPE_FIXTURE_DIR}/overflow.lwm")
    add_custom_command(
        OUTPUT ${LW_RESHAPE_FIXTURES}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_reshape_test_models.py"
                --output-dir "${LW_RESHAPE_FIXTURE_DIR}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_reshape_test_models.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/converter/lwm_v0.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Creating direct dynamic Reshape contract fixtures"
        VERBATIM)
    add_custom_target(dynamic_reshape_fixtures DEPENDS ${LW_RESHAPE_FIXTURES})
    add_executable(test-dynamic-reshape tests/reshape_dynamic_driver.c)
    target_link_libraries(test-dynamic-reshape PRIVATE lw_ppocr_c)
    target_compile_features(test-dynamic-reshape PRIVATE c_std_11)
    add_dependencies(test-dynamic-reshape dynamic_reshape_fixtures)
    if(MSVC)
        target_compile_options(test-dynamic-reshape PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-dynamic-reshape PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME dynamic_reshape_contract
        COMMAND test-dynamic-reshape ${LW_RESHAPE_FIXTURES})

    add_test(
        NAME model_analysis
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_analyze_onnx tests.test_compare_model_analysis
                tests.test_cls_identity tests.test_model_contract
                tests.test_small_validation_bundle
                tests.test_model_catalog
                tests.test_model_manifest
                tests.test_probe_rec_shape_metadata
                tests.test_small_contracts
                tests.test_medium_contracts
                tests.test_ocr_dataset
                tests.test_generate_ocr_dataset
                tests.test_prepare_ocr_dataset_ppm
                tests.test_build_ocr_width_switch_manifest
                tests.test_compare_ocr_dataset_runtime
                tests.test_collect_rec_shape_profile
                tests.test_collect_det_shape_profile
                tests.test_compare_ocr_dataset_reports
                tests.test_build_engine_comparison_report
                tests.test_layout_plan_snapshot
                tests.test_summarize_ocr_dataset_reports
                tests.test_check_ocr_dataset_regression
                tests.test_experimental_ops tests.test_small_ocr_baseline
                tests.test_runtime_model_pack
                tests.test_release_assets
                tests.test_web_ocr_contracts
                tests.test_versioning)
    set_tests_properties(model_analysis PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME release_metadata_consistency
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_release_readiness.py"
                --mode preview
                --version "${PROJECT_VERSION}")
    set_tests_properties(release_metadata_consistency PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME wasm_host_abi_contract
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_web_abi_contract)
    set_tests_properties(wasm_host_abi_contract PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME conv_shape_analysis
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_analyze_conv_shapes)
    set_tests_properties(conv_shape_analysis PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME rec_aot_pattern_analysis
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_analyze_rec_aot_patterns)
    set_tests_properties(rec_aot_pattern_analysis PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME ocr_dispatch_profile_comparison
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_compare_ocr_dispatch_profiles)
    set_tests_properties(ocr_dispatch_profile_comparison PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME rec_runtime_profile_comparison
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_compare_rec_runtime_profiles)
    set_tests_properties(rec_runtime_profile_comparison PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(
        NAME opencv_dnn_comparison_helpers
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_compare_opencv_dnn)
    set_tests_properties(opencv_dnn_comparison_helpers PROPERTIES
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

    add_executable(test-crop-buffer tests/test_crop_buffer.c)
    target_link_libraries(test-crop-buffer PRIVATE lw_ppocr_c)
    target_include_directories(test-crop-buffer PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(test-crop-buffer PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(test-crop-buffer PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-crop-buffer PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME crop_buffer COMMAND test-crop-buffer)
    add_executable(test-session-planner tests/test_session.c)
    target_link_libraries(test-session-planner PRIVATE lw_ppocr_c)
    target_include_directories(test-session-planner PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    if(LW_EXPERIMENTAL_PREPARED_EXECUTION)
        target_compile_definitions(test-session-planner PRIVATE LW_EXPERIMENTAL_PREPARED_EXECUTION=1)
    endif()
    target_compile_features(test-session-planner PRIVATE c_std_11)
    add_dependencies(test-session-planner convert_rec_model)
    if(MSVC)
        target_compile_options(test-session-planner PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-session-planner PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME session_planner COMMAND test-session-planner "${LW_PPOCR_C_REC_LWM}")
    add_executable(layout-plan-driver
        tests/layout_plan_driver.c
        src/runtime/layout_planner.c)
    target_link_libraries(layout-plan-driver PRIVATE lw_ppocr_c)
    target_include_directories(layout-plan-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(layout-plan-driver PRIVATE c_std_11)
    add_dependencies(layout-plan-driver convert_rec_model convert_det_model)
    if(MSVC)
        target_compile_options(layout-plan-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(layout-plan-driver PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME layout_plan_analysis
        COMMAND layout-plan-driver "${LW_PPOCR_C_REC_LWM}" 48 960)
    add_test(NAME layout_plan_direct_input
        COMMAND layout-plan-driver "${LW_PPOCR_C_REC_LWM}" 48 960 direct-nhwc)
    add_test(NAME layout_plan_det
        COMMAND layout-plan-driver "${LW_PPOCR_C_DET_LWM}" 640 640 direct-nhwc)

    add_executable(workspace-report-driver tests/workspace_report_driver.c)
    target_link_libraries(workspace-report-driver PRIVATE lw_ppocr_c)
    target_include_directories(workspace-report-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(workspace-report-driver PRIVATE c_std_11)
    add_dependencies(workspace-report-driver convert_rec_model)
    if(MSVC)
        target_compile_options(workspace-report-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(workspace-report-driver PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    if(LW_EXPERIMENTAL_CTC_TILED)
        target_compile_definitions(workspace-report-driver PRIVATE LW_EXPERIMENTAL_CTC_TILED=1)
    endif()
    add_test(NAME workspace_memory_report
        COMMAND workspace-report-driver "${LW_PPOCR_C_REC_LWM}" 320 1)
    add_test(NAME workspace_ctc_memory_report
        COMMAND workspace-report-driver "${LW_PPOCR_C_REC_LWM}" 320 1 --ctc-greedy)

    add_executable(test-cpu-topology tests/test_cpu_topology.c)
    target_link_libraries(test-cpu-topology PRIVATE lw_ppocr_c)
    target_compile_definitions(test-cpu-topology PRIVATE
        LW_X64_OCR_THREAD_BUDGET=${LW_X64_OCR_THREAD_BUDGET})
    target_include_directories(test-cpu-topology PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(test-cpu-topology PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(test-cpu-topology PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-cpu-topology PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME cpu_topology_policy COMMAND test-cpu-topology)

    add_executable(test-c-abi tests/test_abi.c)
    target_link_libraries(test-c-abi PRIVATE lw_ppocr_c)
    target_compile_features(test-c-abi PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(test-c-abi PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-c-abi PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME c_abi_contract COMMAND test-c-abi)

    add_executable(test-c-abi-prefix tests/test_abi_prefix.c)
    target_link_libraries(test-c-abi-prefix PRIVATE lw_ppocr_c)
    target_compile_features(test-c-abi-prefix PRIVATE c_std_11)
    add_dependencies(test-c-abi-prefix convert_det_model convert_cls_model convert_rec_model)
    if(MSVC)
        target_compile_options(test-c-abi-prefix PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(test-c-abi-prefix PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME c_abi_prefix_compatibility
        COMMAND test-c-abi-prefix
                "${LW_PPOCR_C_DET_LWM}"
                "${LW_PPOCR_C_CLS_LWM}"
                "${LW_PPOCR_C_REC_LWM}"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt")

    add_executable(abi-layout-driver tests/abi_layout_driver.c)
    target_include_directories(abi-layout-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(abi-layout-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(abi-layout-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(abi-layout-driver PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME abi_layout_contract
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_abi_layout.py"
                --driver "$<TARGET_FILE:abi-layout-driver>"
                --manifest "${CMAKE_CURRENT_SOURCE_DIR}/abi/c-abi-v1-layout.json")

    add_test(
        NAME lwm_layout_contract
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lwm_layout.py"
                --manifest "${CMAKE_CURRENT_SOURCE_DIR}/abi/lwm-v0.1-layout.json"
                --model "${LW_PPOCR_C_DET_LWM}"
                --model "${LW_PPOCR_C_CLS_LWM}"
                --model "${LW_PPOCR_C_REC_LWM}")
    set_tests_properties(lwm_layout_contract PROPERTIES
        DEPENDS "det_converter;cls_converter;rec_converter")

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

    add_test(
        NAME shared_export_candidate_contract
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_abi_v1_candidate.py"
                --header "${CMAKE_CURRENT_SOURCE_DIR}/include/lw_infer.h"
                --manifest "${CMAKE_CURRENT_SOURCE_DIR}/abi/c-abi-v1-candidate.json"
                --allowlist "${CMAKE_CURRENT_SOURCE_DIR}/abi/exports-v1-candidate.txt"
                --library "$<TARGET_FILE:lw_ppocr_c_shared>"
                --tool "${LW_EXPORT_TOOL}")
    set_tests_properties(shared_export_candidate_contract PROPERTIES TIMEOUT 60)

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

    add_executable(conv3x3-benchmark-driver
        tests/conv3x3_benchmark_driver.c)
    target_link_libraries(conv3x3-benchmark-driver PRIVATE lw_ppocr_c)
    target_include_directories(conv3x3-benchmark-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(conv3x3-benchmark-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(conv3x3-benchmark-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(conv3x3-benchmark-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME conv3x3_benchmark_smoke
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv3x3_benchmark.py"
                --driver "$<TARGET_FILE:conv3x3-benchmark-driver>")
    set_tests_properties(conv3x3_benchmark_smoke PROPERTIES TIMEOUT 120)

    add_executable(convtranspose-benchmark-driver
        tests/convtranspose_benchmark_driver.c)
    target_link_libraries(convtranspose-benchmark-driver PRIVATE lw_ppocr_c)
    target_include_directories(convtranspose-benchmark-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(convtranspose-benchmark-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(convtranspose-benchmark-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(convtranspose-benchmark-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME convtranspose_benchmark_smoke
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_convtranspose_benchmark.py"
                --driver "$<TARGET_FILE:convtranspose-benchmark-driver>")
    set_tests_properties(convtranspose_benchmark_smoke PROPERTIES TIMEOUT 120)

    add_executable(packed-conv1x1-benchmark-driver
        tests/packed_conv1x1_benchmark_driver.c)
    target_link_libraries(packed-conv1x1-benchmark-driver PRIVATE lw_ppocr_c)
    target_include_directories(packed-conv1x1-benchmark-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(packed-conv1x1-benchmark-driver PRIVATE c_std_11)
    if(LW_EXPERIMENTAL_AVX2_FMA_DISPATCH OR LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH)
        target_compile_definitions(packed-conv1x1-benchmark-driver PRIVATE LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH=1)
    endif()
    if(MSVC)
        target_compile_options(packed-conv1x1-benchmark-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(packed-conv1x1-benchmark-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME packed_conv1x1_benchmark_smoke
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packed_conv1x1_benchmark.py"
                --driver "$<TARGET_FILE:packed-conv1x1-benchmark-driver>")
    set_tests_properties(packed_conv1x1_benchmark_smoke PROPERTIES TIMEOUT 120)

    add_executable(packed-matmul-benchmark-driver
        tests/packed_matmul_benchmark_driver.c)
    target_link_libraries(packed-matmul-benchmark-driver PRIVATE lw_ppocr_c)
    target_include_directories(packed-matmul-benchmark-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(packed-matmul-benchmark-driver PRIVATE c_std_11)
    if(LW_EXPERIMENTAL_AVX2_FMA_DISPATCH OR LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH)
        target_compile_definitions(packed-matmul-benchmark-driver PRIVATE LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH=1)
    endif()
    if(MSVC)
        target_compile_options(packed-matmul-benchmark-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(packed-matmul-benchmark-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME packed_matmul_benchmark_smoke
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packed_matmul_benchmark.py"
                --driver "$<TARGET_FILE:packed-matmul-benchmark-driver>")
    set_tests_properties(packed_matmul_benchmark_smoke PROPERTIES TIMEOUT 120)
    add_executable(erf-benchmark-driver
        tests/erf_benchmark_driver.c)
    target_link_libraries(erf-benchmark-driver PRIVATE lw_ppocr_c)
    target_include_directories(erf-benchmark-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(erf-benchmark-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(erf-benchmark-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(erf-benchmark-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME erf_benchmark_smoke
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_erf_benchmark.py"
                --driver "$<TARGET_FILE:erf-benchmark-driver>")
    set_tests_properties(erf_benchmark_smoke PROPERTIES TIMEOUT 120)

    add_executable(conv3x3-stride2-benchmark-driver
        tests/conv3x3_stride2_benchmark_driver.c)
    target_link_libraries(conv3x3-stride2-benchmark-driver PRIVATE lw_ppocr_c)
    target_include_directories(conv3x3-stride2-benchmark-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(conv3x3-stride2-benchmark-driver PRIVATE c_std_11)
    if(LW_EXPERIMENTAL_AVX2_FMA_DISPATCH)
        target_compile_definitions(conv3x3-stride2-benchmark-driver PRIVATE LW_EXPERIMENTAL_AVX2_FMA_DISPATCH=1)
    endif()
    if(LW_AVX2_FMA_CONV3X3_DISPATCH AND NOT EMSCRIPTEN)
        target_compile_definitions(conv3x3-stride2-benchmark-driver PRIVATE LW_AVX2_FMA_CONV3X3_DISPATCH=1)
    endif()
    if(MSVC)
        target_compile_options(conv3x3-stride2-benchmark-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(conv3x3-stride2-benchmark-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME conv3x3_stride2_benchmark_smoke
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv3x3_stride2_benchmark.py"
                --driver "$<TARGET_FILE:conv3x3-stride2-benchmark-driver>")
    set_tests_properties(conv3x3_stride2_benchmark_smoke PROPERTIES TIMEOUT 120)
    set_tests_properties(conv_kernel_reference PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(simd-backend-driver tests/simd_backend_driver.c)
    target_link_libraries(simd-backend-driver PRIVATE lw_ppocr_c)
    target_include_directories(simd-backend-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(simd-backend-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(simd-backend-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(simd-backend-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME simd_backend_detection COMMAND simd-backend-driver)

    add_executable(rec-graph-driver tests/rec_graph_driver.c)
    target_link_libraries(rec-graph-driver PRIVATE lw_ppocr_c)
    target_include_directories(rec-graph-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
    target_compile_features(rec-graph-driver PRIVATE c_std_11)
    add_dependencies(rec-graph-driver convert_rec_model)
    if(MSVC)
        target_compile_definitions(rec-graph-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
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
    set(LW_ONNX_WIDTH_TEST_ARGS)
    if(LW_NATIVE_FINE_REC_WIDTHS OR LW_ARM64_FINE_REC_WIDTHS)
        list(APPEND LW_ONNX_WIDTH_TEST_ARGS --fine-rec-widths)
    endif()
    add_test(NAME native_onnx_import
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_onnx_import.py"
            --inspect "$<TARGET_FILE:lwm-inspect>"
            --rec-driver "$<TARGET_FILE:rec-graph-driver>"
            --ocr-driver "$<TARGET_FILE:lw-ocr-ppm>"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --sample "${LW_PPOCR_C_DET_DEMO_PPM}"
            ${LW_ONNX_WIDTH_TEST_ARGS}
            --report "${CMAKE_CURRENT_BINARY_DIR}/onnx-import-report.json")
    set_tests_properties(native_onnx_import PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 600)
    add_test(NAME deployment_model_formats
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v tests.test_deployment_model_formats)
    set_tests_properties(deployment_model_formats PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 120)

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
        target_compile_definitions(det-graph-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
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
        target_compile_definitions(det-pipeline-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
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

    add_executable(reading-order-driver tests/reading_order_driver.c)
    target_link_libraries(reading-order-driver PRIVATE lw_ppocr_c)
    target_include_directories(reading-order-driver PRIVATE src/ppocr)
    target_compile_features(reading-order-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_options(reading-order-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(reading-order-driver PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME reading_order_sorting COMMAND reading-order-driver)

    add_executable(ocr-pipeline-driver tests/ocr_pipeline_driver.c)
    target_link_libraries(ocr-pipeline-driver PRIVATE lw_ppocr_c)
    target_include_directories(ocr-pipeline-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(ocr-pipeline-driver PRIVATE c_std_11)
    add_dependencies(ocr-pipeline-driver convert_det_model convert_cls_model
        convert_rec_model)
    if(MSVC)
        target_compile_definitions(ocr-pipeline-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
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
    set(full_ocr_profile_test_args
        --driver "$<TARGET_FILE:full-ocr-profile-driver>"
        --det-model "${LW_PPOCR_C_DET_LWM}"
        --cls-model "${LW_PPOCR_C_CLS_LWM}"
        --rec-model "${LW_PPOCR_C_REC_LWM}"
        --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
        --image "${LW_PPOCR_C_DET_DEMO_PPM}"
        --thread-budget "${LW_X64_OCR_THREAD_BUDGET}")
    if(LW_REC_RESIDENT_WIDTHS AND NOT EMSCRIPTEN)
        list(APPEND full_ocr_profile_test_args --expect-resident)
    endif()
    if(LW_X64_REC_FAST_ENABLED)
        list(APPEND full_ocr_profile_test_args --expect-compiled-rec)
    endif()
    if(LW_EXPERIMENTAL_PREPARED_EXECUTION AND NOT EMSCRIPTEN)
        list(APPEND full_ocr_profile_test_args --expect-prepared)
    endif()
    add_test(
        NAME full_ocr_operator_profile
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_full_ocr_profile.py"
                ${full_ocr_profile_test_args})
    set_tests_properties(full_ocr_operator_profile PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        TIMEOUT 360)

    # Test-only benchmark variant: it links the static runtime so an internal
    # DET intra-op override can be measured without expanding the public ABI.
    add_executable(full-ocr-intra-benchmark examples/ocr_benchmark_ppm.c)
    target_link_libraries(full-ocr-intra-benchmark PRIVATE lw_ppocr_c)
    target_compile_definitions(full-ocr-intra-benchmark PRIVATE
        LW_OCR_BENCHMARK_INTERNAL=1)
    target_include_directories(full-ocr-intra-benchmark PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(full-ocr-intra-benchmark PRIVATE c_std_11)
    add_dependencies(full-ocr-intra-benchmark convert_det_model convert_cls_model
        convert_rec_model det_demo_fixture)
    if(WIN32)
        target_link_libraries(full-ocr-intra-benchmark PRIVATE shell32 psapi)
    endif()
    if(MSVC)
        target_compile_options(full-ocr-intra-benchmark PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(full-ocr-intra-benchmark PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()

    # Same-process dataset benchmark used to measure resident/compact memory
    # and latency without restarting the OCR runtime for every image.
    if(NOT EMSCRIPTEN)
        add_executable(full-ocr-dataset-benchmark examples/ocr_dataset_benchmark.c)
        target_link_libraries(full-ocr-dataset-benchmark PRIVATE lw_ppocr_c)
        target_include_directories(full-ocr-dataset-benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/examples"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
        target_compile_features(full-ocr-dataset-benchmark PRIVATE c_std_11)
        add_dependencies(full-ocr-dataset-benchmark convert_det_model convert_cls_model
            convert_rec_model det_demo_fixture)
        if(LW_REC_RESIDENT_WIDTHS)
            target_compile_definitions(full-ocr-dataset-benchmark PRIVATE LW_REC_RESIDENT_WIDTHS=1)
        endif()
        if(WIN32)
            target_link_libraries(full-ocr-dataset-benchmark PRIVATE shell32 psapi)
        endif()
        if(MSVC)
            target_compile_definitions(full-ocr-dataset-benchmark PRIVATE _CRT_SECURE_NO_WARNINGS)
            target_compile_options(full-ocr-dataset-benchmark PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(full-ocr-dataset-benchmark PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        if(LW_ADAPTIVE_REC_BUDGET_ENABLED)
            add_test(
                NAME full_ocr_adaptive_rec_budget
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ocr_adaptive_rec_budget.py"
                        --benchmark "$<TARGET_FILE:full-ocr-dataset-benchmark>"
                        --det "${LW_PPOCR_C_DET_LWM}"
                        --cls "${LW_PPOCR_C_CLS_LWM}"
                        --rec "${LW_PPOCR_C_REC_LWM}"
                        --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                        --sample "${LW_PPOCR_C_DET_DEMO_PPM}")
            set_tests_properties(full_ocr_adaptive_rec_budget PROPERTIES TIMEOUT 180)
        endif()
    endif()

    add_executable(rec-preprocess-layout-driver tests/rec_preprocess_layout_driver.c)
    target_link_libraries(rec-preprocess-layout-driver PRIVATE lw_ppocr_c)
    target_include_directories(rec-preprocess-layout-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(rec-preprocess-layout-driver PRIVATE c_std_11)
    if(MSVC)
        target_compile_definitions(rec-preprocess-layout-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(rec-preprocess-layout-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(rec-preprocess-layout-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME rec_preprocess_layout COMMAND rec-preprocess-layout-driver)
    set_tests_properties(rec_preprocess_layout PROPERTIES TIMEOUT 30)
    add_executable(nhwc-conv1x1-driver tests/nhwc_conv1x1_driver.c)
    target_link_libraries(nhwc-conv1x1-driver PRIVATE lw_ppocr_c)
    target_include_directories(nhwc-conv1x1-driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
    target_compile_features(nhwc-conv1x1-driver PRIVATE c_std_11)
    if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "(AMD64|x86_64|amd64|X86)")
        target_compile_options(nhwc-conv1x1-driver PRIVATE /arch:AVX2)
    endif()
    if(MSVC)
        target_compile_definitions(nhwc-conv1x1-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(nhwc-conv1x1-driver PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(nhwc-conv1x1-driver PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME nhwc_conv1x1_layout COMMAND nhwc-conv1x1-driver)
    set_tests_properties(nhwc_conv1x1_layout PROPERTIES TIMEOUT 30)
    include(cmake/X64Tests.cmake)
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
        target_compile_definitions(rec-pipeline-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
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
