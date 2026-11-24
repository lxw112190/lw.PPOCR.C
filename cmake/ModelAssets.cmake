# Model analysis and the bundled Tiny model conversion targets.

if(NOT LW_RUNTIME_ONLY)
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
if(LW_DEPLOY_ONNX_MODELS)
    set(LW_DEPLOY_MODEL_FORMAT onnx)
    set(LW_DEPLOY_DET "${LW_PPOCR_C_DET_ONNX}")
    set(LW_DEPLOY_CLS "${LW_PPOCR_C_CLS_ONNX}")
    set(LW_DEPLOY_REC "${LW_PPOCR_C_REC_ONNX}")
else()
    set(LW_DEPLOY_MODEL_FORMAT lwm)
    set(LW_DEPLOY_DET "${LW_PPOCR_C_DET_LWM}")
    set(LW_DEPLOY_CLS "${LW_PPOCR_C_CLS_LWM}")
    set(LW_DEPLOY_REC "${LW_PPOCR_C_REC_LWM}")
endif()

add_custom_target(analyze_models
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/converter/analyze_onnx.py"
            --json-output "${LW_PPOCR_C_ANALYSIS_JSON}"
            --markdown-output "${LW_PPOCR_C_ANALYSIS_MARKDOWN}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Analyzing bundled PP-OCRv6 tiny ONNX models"
    VERBATIM)

if(LW_PRECONVERTED_ASSETS_DIR)
    get_filename_component(LW_PRECONVERTED_ASSETS_DIR
        "${LW_PRECONVERTED_ASSETS_DIR}" ABSOLUTE)
    foreach(asset IN ITEMS
            rec.lwm rec-lwm.json cls.lwm cls-lwm.json det.lwm det-lwm.json
            sample-crop.ppm sample.ppm)
        if(NOT EXISTS "${LW_PRECONVERTED_ASSETS_DIR}/${asset}")
            message(FATAL_ERROR
                "LW_PRECONVERTED_ASSETS_DIR is missing required file: ${asset}")
        endif()
    endforeach()
    message(STATUS
        "Using preconverted models and fixtures from ${LW_PRECONVERTED_ASSETS_DIR}")

    function(lw_copy_preconverted_pair output_a name_a output_b name_b label)
        add_custom_command(
            OUTPUT "${output_a}" "${output_b}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/models"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${LW_PRECONVERTED_ASSETS_DIR}/${name_a}" "${output_a}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${LW_PRECONVERTED_ASSETS_DIR}/${name_b}" "${output_b}"
            DEPENDS
                "${LW_PRECONVERTED_ASSETS_DIR}/${name_a}"
                "${LW_PRECONVERTED_ASSETS_DIR}/${name_b}"
            COMMENT "Copying preconverted ${label} assets"
            VERBATIM)
    endfunction()

    lw_copy_preconverted_pair(
        "${LW_PPOCR_C_REC_LWM}" rec.lwm
        "${LW_PPOCR_C_REC_LWM_METADATA}" rec-lwm.json REC)
    lw_copy_preconverted_pair(
        "${LW_PPOCR_C_CLS_LWM}" cls.lwm
        "${LW_PPOCR_C_CLS_LWM_METADATA}" cls-lwm.json CLS)
    lw_copy_preconverted_pair(
        "${LW_PPOCR_C_DET_LWM}" det.lwm
        "${LW_PPOCR_C_DET_LWM_METADATA}" det-lwm.json DET)
else()
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
endif()

add_custom_target(convert_rec_model ALL
    DEPENDS "${LW_PPOCR_C_REC_LWM}" "${LW_PPOCR_C_REC_LWM_METADATA}")
add_custom_target(convert_cls_model ALL
    DEPENDS "${LW_PPOCR_C_CLS_LWM}" "${LW_PPOCR_C_CLS_LWM_METADATA}")
add_custom_target(convert_det_model ALL
    DEPENDS "${LW_PPOCR_C_DET_LWM}" "${LW_PPOCR_C_DET_LWM_METADATA}")
endif()
