cmake_minimum_required(VERSION 3.20)

project(lw.PPOCR.C VERSION 0.1.0 LANGUAGES C)

include(CTest)
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

add_library(lw_ppocr_c STATIC
    src/kernels/scalar_activation.c
    src/kernels/scalar_conv.c
    src/kernels/scalar_elementwise.c
    src/kernels/scalar_layout.c
    src/kernels/scalar_matmul.c
    src/kernels/scalar_reduction.c
    src/kernels/scalar_softmax.c
    src/runtime/memory.c
    src/runtime/model.c
    src/runtime/session.c
    src/runtime/shape.c
    src/runtime/validate.c)
target_include_directories(lw_ppocr_c
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
target_compile_features(lw_ppocr_c PUBLIC c_std_11)
if(MSVC)
    target_compile_options(lw_ppocr_c PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw_ppocr_c PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(lw_ppocr_c PUBLIC m)
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

    add_test(
        NAME lwm_corruption
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lwm_corruption.py"
                --inspect "$<TARGET_FILE:lwm-inspect>"
                --model "${LW_PPOCR_C_REC_LWM}")
    set_tests_properties(lwm_corruption PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
