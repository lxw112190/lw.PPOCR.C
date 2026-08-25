cmake_minimum_required(VERSION 3.20)

project(lw.PPOCR.C VERSION 0.1.0 LANGUAGES C)

include(CTest)
find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

set(LW_PPOCR_C_ANALYSIS_JSON
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/ppocrv6-tiny-analysis.json")
set(LW_PPOCR_C_ANALYSIS_MARKDOWN
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/SUPPORTED_OPS_V0.md")

add_custom_target(analyze_models
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/converter/analyze_onnx.py"
            --json-output "${LW_PPOCR_C_ANALYSIS_JSON}"
            --markdown-output "${LW_PPOCR_C_ANALYSIS_MARKDOWN}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Analyzing bundled PP-OCRv6 tiny ONNX models"
    VERBATIM)

if(BUILD_TESTING)
    add_test(
        NAME model_analysis
        COMMAND "${Python3_EXECUTABLE}" -m unittest -v
                tests.test_analyze_onnx)
    set_tests_properties(model_analysis PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
