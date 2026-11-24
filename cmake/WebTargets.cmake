# Browser and Node WASM products, packaging, and WebAssembly contract drivers.

if(EMSCRIPTEN)
    if(LW_WEB_LEGACY)
        find_program(LW_NODE_EXECUTABLE NAMES node nodejs REQUIRED)
        set(LW_LEGACY_ESBUILD_ENTRY
            "${CMAKE_CURRENT_SOURCE_DIR}/web/node_modules/esbuild/package.json")
        if(NOT EXISTS "${LW_LEGACY_ESBUILD_ENTRY}")
            message(FATAL_ERROR
                "LW_WEB_LEGACY requires web/node_modules; run npm ci in web first")
        endif()
    endif()
    add_executable(lw_ppocr_web web/lw_web_api.c)
    target_link_libraries(lw_ppocr_web PRIVATE lw_ppocr_c)
    target_include_directories(lw_ppocr_web PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(lw_ppocr_web PRIVATE c_std_11)
    target_compile_options(lw_ppocr_web PRIVATE -O3 -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(lw_ppocr_web PRIVATE
        LW_WEB_ABI_VERSION=${LW_WASM_HOST_ABI_VERSION})
    if(LW_WASM_SIMD128)
        target_compile_options(lw_ppocr_web PRIVATE -msimd128 -msse2)
        target_link_options(lw_ppocr_web PRIVATE -msimd128)
    endif()
    if(LW_WEB_LEGACY)
        target_link_options(lw_ppocr_web PRIVATE
            "-sMIN_CHROME_VERSION=${LW_WEB_MIN_CHROME_VERSION}"
            "-sENVIRONMENT=web,webview,worker")
    endif()
    target_link_options(lw_ppocr_web PRIVATE
        -O3
        "-sMODULARIZE=1" "-sEXPORT_NAME=LwPpocrModule" "-sSINGLE_FILE=1"
        "-sFORCE_FILESYSTEM=1" "-sALLOW_MEMORY_GROWTH=1" "-sNO_EXIT_RUNTIME=1"
        "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8','HEAPU32','HEAPF32']"
        "-sEXPORTED_FUNCTIONS=['_lw_web_init','_lw_web_shutdown','_lw_web_info_size','_lw_web_get_info','_lw_web_set_reading_order','_lw_web_run','_lw_web_last_error','_lw_web_malloc','_lw_web_free']"
        "--no-entry")
    set_target_properties(lw_ppocr_web PROPERTIES OUTPUT_NAME "lw_ppocr_web")
    if(NOT LW_DEPLOY_ONNX_MODELS)
        add_dependencies(lw_ppocr_web convert_det_model convert_cls_model convert_rec_model)
    endif()

    # The generated runtime remains a normal Emscripten module. The browser
    # SDK is a separate deterministic artifact layered on top of that module,
    # so other applications can consume it without the demo HTML.
    if(LW_WEB_LEGACY)
        set(LW_WEB_OUTPUT_SUFFIX "-legacy")
        set(LW_WEB_BUILD_FLAVOR "legacy")
        set(LW_WEB_WASM_SIMD128 "false")
        set(LW_WEB_PDF_ENABLED "false")
        set(LW_WEB_MIN_CHROME_VERSION_VALUE "${LW_WEB_MIN_CHROME_VERSION}")
        set(LW_WEB_JS_TARGET_VALUE "chrome${LW_WEB_MIN_CHROME_VERSION}")
        set(LW_WEB_HTML_TITLE
            "lw.PPOCR.C · PP-OCRv6 Tiny · 兼容版离线 OCR 工作台")
    else()
        set(LW_WEB_OUTPUT_SUFFIX "")
        set(LW_WEB_BUILD_FLAVOR "modern")
        if(LW_WASM_SIMD128)
            set(LW_WEB_WASM_SIMD128 "true")
        else()
            set(LW_WEB_WASM_SIMD128 "false")
        endif()
        if(LW_WEB_PDF)
            set(LW_WEB_PDF_ENABLED "true")
        else()
            set(LW_WEB_PDF_ENABLED "false")
        endif()
        set(LW_WEB_MIN_CHROME_VERSION_VALUE "")
        set(LW_WEB_JS_TARGET_VALUE "")
        set(LW_WEB_HTML_TITLE
            "lw.PPOCR.C · PP-OCRv6 Tiny · 离线 OCR 工作台")
    endif()

    if(LW_WEB_LEGACY)
        set(LW_PPOCR_C_LEGACY_RUNTIME_JS
            "${CMAKE_CURRENT_BINARY_DIR}/lw_ppocr_web.legacy.js")
        set(LW_PPOCR_C_LEGACY_RUNTIME_REPORT
            "${CMAKE_CURRENT_BINARY_DIR}/legacy-runtime-report.json")
        add_custom_command(
            OUTPUT
                "${LW_PPOCR_C_LEGACY_RUNTIME_JS}"
                "${LW_PPOCR_C_LEGACY_RUNTIME_REPORT}"
            COMMAND "${LW_NODE_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/web/transpile_legacy_runtime.mjs"
                    --input "$<TARGET_FILE_DIR:lw_ppocr_web>/lw_ppocr_web.js"
                    --output "${LW_PPOCR_C_LEGACY_RUNTIME_JS}"
                    --target "chrome${LW_WEB_MIN_CHROME_VERSION}"
                    --report "${LW_PPOCR_C_LEGACY_RUNTIME_REPORT}"
            DEPENDS
                lw_ppocr_web
                "${CMAKE_CURRENT_SOURCE_DIR}/web/transpile_legacy_runtime.mjs"
                "${CMAKE_CURRENT_SOURCE_DIR}/web/package-lock.json"
                "${CMAKE_CURRENT_SOURCE_DIR}/web/node_modules/esbuild/package.json"
            COMMENT "Lowering Legacy Web runtime for Chrome ${LW_WEB_MIN_CHROME_VERSION}"
            VERBATIM)
        add_custom_target(lw-legacy-runtime
            DEPENDS
                "${LW_PPOCR_C_LEGACY_RUNTIME_JS}"
                "${LW_PPOCR_C_LEGACY_RUNTIME_REPORT}")
        set(LW_PPOCR_C_WEB_RUNTIME_FOR_PACKAGE
            "${LW_PPOCR_C_LEGACY_RUNTIME_JS}")
        set(LW_PPOCR_C_WEB_RUNTIME_DEPENDS lw-legacy-runtime)
    else()
        set(LW_PPOCR_C_WEB_RUNTIME_FOR_PACKAGE
            "$<TARGET_FILE_DIR:lw_ppocr_web>/lw_ppocr_web.js")
        set(LW_PPOCR_C_WEB_RUNTIME_DEPENDS)
    endif()
    set(LW_PPOCR_C_WEB_SDK
        "${CMAKE_CURRENT_BINARY_DIR}/lw-ppocr${LW_WEB_OUTPUT_SUFFIX}.js")
    add_custom_command(
        OUTPUT "${LW_PPOCR_C_WEB_SDK}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_sdk.py"
            --template "${CMAKE_CURRENT_SOURCE_DIR}/web/lw_ppocr_sdk.template.js"
            --runtime "${LW_PPOCR_C_WEB_RUNTIME_FOR_PACKAGE}"
            --det "${LW_DEPLOY_DET}" --cls "${LW_DEPLOY_CLS}"
            --rec "${LW_DEPLOY_REC}"
            --dictionary "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
            --model-family "PP-OCRv6"
            --model-variant "tiny"
            --model-display-name "PP-OCRv6 Tiny"
            --version "${PROJECT_VERSION}"
            --build-flavor "${LW_WEB_BUILD_FLAVOR}"
            --wasm-simd128 "${LW_WEB_WASM_SIMD128}"
            --pdf-enabled "${LW_WEB_PDF_ENABLED}"
            --min-chrome-version "${LW_WEB_MIN_CHROME_VERSION_VALUE}"
            --js-target "${LW_WEB_JS_TARGET_VALUE}"
            --output "${LW_PPOCR_C_WEB_SDK}"
        DEPENDS
            lw_ppocr_web
            ${LW_PPOCR_C_WEB_RUNTIME_DEPENDS}
            "${CMAKE_CURRENT_SOURCE_DIR}/web/lw_ppocr_sdk.template.js"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_sdk.py"
            "${LW_DEPLOY_DET}" "${LW_DEPLOY_CLS}" "${LW_DEPLOY_REC}"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
        COMMENT "Packaging the browser OCR SDK"
        VERBATIM)
    add_custom_target(lw-ocr-js DEPENDS "${LW_PPOCR_C_WEB_SDK}")

    # The standalone HTML is now only a thin UI shell plus the SDK. Keep it as
    # a separate target so CI and downstream users can download either file.
    set(LW_PPOCR_C_WEB_HTML
        "${CMAKE_CURRENT_BINARY_DIR}/ocr-demo${LW_WEB_OUTPUT_SUFFIX}.html")
    set(LW_PPOCR_C_WEB_HTML_PDF_ARGS --no-pdf)
    set(LW_PPOCR_C_WEB_HTML_PDF_DEPENDS)
    if(LW_WEB_PDF)
        set(LW_PPOCR_C_WEB_HTML_PDF_ARGS
            --pdf-adapter "${CMAKE_CURRENT_SOURCE_DIR}/web/pdf/lw_pdf_adapter.js"
            --pdfjs-core "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/pdf.min.mjs"
            --pdfjs-worker "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/pdf.worker.min.mjs"
            --pdfjs-jbig2 "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/jbig2.wasm"
            --pdfjs-openjpeg "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/openjpeg.wasm"
            --pdfjs-qcms "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/qcms_bg.wasm"
            --pdfjs-version "6.3.289")
        set(LW_PPOCR_C_WEB_HTML_PDF_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/web/pdf/lw_pdf_adapter.js"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/pdf.min.mjs"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/pdf.worker.min.mjs"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/LICENSE"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/VERSION"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/jbig2.wasm"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/openjpeg.wasm"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/qcms_bg.wasm"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/LICENSE_JBIG2"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/LICENSE_OPENJPEG"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/LICENSE_PDFJS_JBIG2"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/LICENSE_PDFJS_OPENJPEG"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/LICENSE_PDFJS_QCMS"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/vendor/pdfjs/LICENSE_QCMS")
    endif()
    add_custom_command(
        OUTPUT "${LW_PPOCR_C_WEB_HTML}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_html.py"
            --template "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo.template.html"
            --sdk "${LW_PPOCR_C_WEB_SDK}"
            --ui "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo-ui.js"
            --sponsor "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/sponsor.jpg"
            --title "${LW_WEB_HTML_TITLE}"
            ${LW_PPOCR_C_WEB_HTML_PDF_ARGS}
            --output "${LW_PPOCR_C_WEB_HTML}"
        DEPENDS
            lw-ocr-js
            "${LW_PPOCR_C_WEB_SDK}"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo.template.html"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo-ui.js"
            "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_html.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/sponsor.jpg"
            ${LW_PPOCR_C_WEB_HTML_PDF_DEPENDS}
        COMMENT "Packaging the standalone offline OCR HTML demo"
        VERBATIM)
    add_custom_target(lw-ocr-html DEPENDS "${LW_PPOCR_C_WEB_HTML}")

    # Tiny remains the compatibility/default browser artifact above.  Small
    # and Medium reuse the same generated WASM runtime and only add their
    # model staging plus SDK/HTML wrappers when explicitly requested.  This
    # keeps the normal browser build fast while providing deterministic,
    # independently downloadable model variants for release jobs.
    if(LW_BUILD_WEB_MODEL_VARIANTS)
        set(LW_WEB_VARIANT_ROOT
            "${CMAKE_CURRENT_BINARY_DIR}/runtime-models")
        set(LW_WEB_VARIANT_COMMON_DEPS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_ppocrv6_runtime_variant.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-models.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-small/det.onnx"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-small/rec.onnx"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-medium/det.onnx"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-medium/rec.onnx"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-shared/PP-OCRv6_small_rec_dict.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_small_det.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_small_rec.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_medium_det.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_medium_rec.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_model_catalog.py")

        function(lw_add_web_model_variant variant display_name)
            set(_variant_conversion_deps)
            if(NOT LW_DEPLOY_ONNX_MODELS)
                set(_variant_conversion_deps convert_cls_model)
            endif()
            set(_variant_root
                "${LW_WEB_VARIANT_ROOT}/ppocrv6-${variant}")
            set(_variant_stage_stamp
                "${_variant_root}/runtime-assets.json")
            add_custom_command(
                OUTPUT "${_variant_stage_stamp}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_ppocrv6_runtime_variant.py"
                        --variant "${variant}"
                        --model-format "${LW_DEPLOY_MODEL_FORMAT}"
                        --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
                        --output-dir "${_variant_root}"
                DEPENDS ${_variant_conversion_deps}
                        ${LW_WEB_VARIANT_COMMON_DEPS}
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                COMMENT "Staging PP-OCRv6 ${display_name} browser model"
                VERBATIM)

            set(_variant_sdk
                "${CMAKE_CURRENT_BINARY_DIR}/lw-ppocr-v6-${variant}.js")
            add_custom_command(
                OUTPUT "${_variant_sdk}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_sdk.py"
                        --template "${CMAKE_CURRENT_SOURCE_DIR}/web/lw_ppocr_sdk.template.js"
                        --runtime "$<TARGET_FILE_DIR:lw_ppocr_web>/lw_ppocr_web.js"
                        --det "${_variant_root}/det.${LW_DEPLOY_MODEL_FORMAT}"
                        --cls "${_variant_root}/cls.${LW_DEPLOY_MODEL_FORMAT}"
                        --rec "${_variant_root}/rec.${LW_DEPLOY_MODEL_FORMAT}"
                        --dictionary "${_variant_root}/ppocr_keys.txt"
                        --model-family "PP-OCRv6"
                        --model-variant "${variant}"
                        --model-display-name "${display_name}"
                        --version "${PROJECT_VERSION}"
                        --build-flavor "${LW_WEB_BUILD_FLAVOR}"
                        --wasm-simd128 "${LW_WEB_WASM_SIMD128}"
                        --pdf-enabled "${LW_WEB_PDF_ENABLED}"
                        --min-chrome-version "${LW_WEB_MIN_CHROME_VERSION_VALUE}"
                        --output "${_variant_sdk}"
                DEPENDS lw_ppocr_web
                        "${_variant_stage_stamp}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/web/lw_ppocr_sdk.template.js"
                        "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_sdk.py"
                COMMENT "Packaging the PP-OCRv6 ${display_name} browser SDK"
                VERBATIM)
            add_custom_target(lw-ocr-js-${variant} DEPENDS "${_variant_sdk}")

            set(_variant_html
                "${CMAKE_CURRENT_BINARY_DIR}/ocr-demo-${variant}.html")
            add_custom_command(
                OUTPUT "${_variant_html}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_html.py"
                        --template "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo.template.html"
                        --sdk "${_variant_sdk}"
                        --ui "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo-ui.js"
                        --sponsor "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/sponsor.jpg"
                        --title "lw.PPOCR.C · ${display_name} · 离线 OCR 工作台"
                        ${LW_PPOCR_C_WEB_HTML_PDF_ARGS}
                        --output "${_variant_html}"
                DEPENDS lw-ocr-js-${variant}
                        "${_variant_sdk}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo.template.html"
                        "${CMAKE_CURRENT_SOURCE_DIR}/web/ocr-demo-ui.js"
                        "${CMAKE_CURRENT_SOURCE_DIR}/web/package_ocr_html.py"
                        "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/sponsor.jpg"
                        ${LW_PPOCR_C_WEB_HTML_PDF_DEPENDS}
                COMMENT "Packaging the PP-OCRv6 ${display_name} offline OCR HTML"
                VERBATIM)
            add_custom_target(lw-ocr-html-${variant} DEPENDS "${_variant_html}")
        endfunction()

        lw_add_web_model_variant(small "PP-OCRv6 Small")
        lw_add_web_model_variant(medium "PP-OCRv6 Medium")
        add_custom_target(lw-ocr-js-all
            DEPENDS lw-ocr-js lw-ocr-js-small lw-ocr-js-medium)
        add_custom_target(lw-ocr-html-all
            DEPENDS lw-ocr-html lw-ocr-html-small lw-ocr-html-medium)
    endif()

    if(NOT LW_WEB_LEGACY)
        # Node consumes the same WASM Host ABI as the browser. Keep this target
    # separate from the browser SDK so the Node release never needs to parse
    # or extract assets from ocr-demo.html.
    add_executable(lw_ppocr_node web/lw_web_api.c)
    target_link_libraries(lw_ppocr_node PRIVATE lw_ppocr_c)
    target_include_directories(lw_ppocr_node PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(lw_ppocr_node PRIVATE c_std_11)
    target_compile_options(lw_ppocr_node PRIVATE -O3 -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(lw_ppocr_node PRIVATE
        LW_WEB_ABI_VERSION=${LW_WASM_HOST_ABI_VERSION})
    if(LW_WASM_SIMD128)
        target_compile_options(lw_ppocr_node PRIVATE -msimd128 -msse2)
        target_link_options(lw_ppocr_node PRIVATE -msimd128)
    endif()
    if(LW_WASM_COMPILED_REC_ENABLED)
        # Only the experimental Node package accepts the C profiler switch.
        # The browser SDK and canonical Node package keep their normal env policy.
        target_include_directories(lw_ppocr_node PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
        target_compile_definitions(lw_ppocr_node PRIVATE LW_NODE_WASM_PROFILE=1)
    endif()
    target_link_options(lw_ppocr_node PRIVATE
        -O3
        "-sMODULARIZE=1" "-sEXPORT_NAME=LwPpocrModule" "-sSINGLE_FILE=1"
        "-sENVIRONMENT=node" "-sFORCE_FILESYSTEM=1" "-sALLOW_MEMORY_GROWTH=1"
        "-sNO_EXIT_RUNTIME=1"
        "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8','HEAPU32','HEAPF32']"
        "-sEXPORTED_FUNCTIONS=['_lw_web_init','_lw_web_shutdown','_lw_web_info_size','_lw_web_get_info','_lw_web_set_reading_order','_lw_web_run','_lw_web_last_error','_lw_web_malloc','_lw_web_free']"
        "--no-entry")
    set_target_properties(lw_ppocr_node PROPERTIES OUTPUT_NAME "lw_ppocr_node")
    if(NOT LW_DEPLOY_ONNX_MODELS)
        add_dependencies(lw_ppocr_node convert_det_model convert_cls_model convert_rec_model)
    endif()

    if(LW_WASM_SIMD128)
        set(LW_NODE_WASM_BACKEND wasm128)
    else()
        set(LW_NODE_WASM_BACKEND scalar)
    endif()

    set(LW_PPOCR_C_NODE_RUNTIME
        "${CMAKE_CURRENT_BINARY_DIR}/runtime.cjs")
    add_custom_command(
        OUTPUT "${LW_PPOCR_C_NODE_RUNTIME}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/node-wasm"
        COMMAND "${CMAKE_COMMAND}" -E copy
                "$<TARGET_FILE:lw_ppocr_node>" "${LW_PPOCR_C_NODE_RUNTIME}"
        DEPENDS lw_ppocr_node
        COMMENT "Preparing the Node.js WASM runtime"
        VERBATIM)
    add_custom_target(lw-node-wasm DEPENDS "${LW_PPOCR_C_NODE_RUNTIME}")

    set(LW_PPOCR_C_NODE_PACKAGE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/node-wasm")
    set(LW_PPOCR_C_NODE_PACKAGE_ARCHIVE
        "${CMAKE_CURRENT_BINARY_DIR}/lw.PPOCR.C-${PROJECT_VERSION}-node-wasm.zip")
    set(LW_PPOCR_C_NODE_PACKAGE_CHECKSUM
        "${LW_PPOCR_C_NODE_PACKAGE_ARCHIVE}.sha256")
    add_custom_command(
        OUTPUT "${LW_PPOCR_C_NODE_PACKAGE_ARCHIVE}"
               "${LW_PPOCR_C_NODE_PACKAGE_CHECKSUM}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/package_node_wasm.py"
                --runtime "${LW_PPOCR_C_NODE_RUNTIME}"
                --det "${LW_DEPLOY_DET}"
                --cls "${LW_DEPLOY_CLS}"
                --rec "${LW_DEPLOY_REC}"
                --dictionary
                "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
                --license "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
                --model-license
                "${CMAKE_CURRENT_SOURCE_DIR}/licenses/PaddleOCR-models-APACHE-2.0.txt"
                --notices "${CMAKE_CURRENT_SOURCE_DIR}/THIRD-PARTY-NOTICES.md"
                --version "${PROJECT_VERSION}"
                --wasm-backend "${LW_NODE_WASM_BACKEND}"
                --wasm-host-abi-version "${LW_WASM_HOST_ABI_VERSION}"
                --lwm-version "${LW_LWM_FORMAT_VERSION}"
                --output-dir "${LW_PPOCR_C_NODE_PACKAGE_DIR}"
                --archive "${LW_PPOCR_C_NODE_PACKAGE_ARCHIVE}"
        DEPENDS lw-node-wasm "${LW_PPOCR_C_NODE_RUNTIME}"
                "${LW_DEPLOY_DET}" "${LW_DEPLOY_CLS}" "${LW_DEPLOY_REC}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/package_node_wasm.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
                "${CMAKE_CURRENT_SOURCE_DIR}/licenses/PaddleOCR-models-APACHE-2.0.txt"
                "${CMAKE_CURRENT_SOURCE_DIR}/THIRD-PARTY-NOTICES.md"
        COMMENT "Packaging the Node.js WASM distribution"
        VERBATIM)
    add_custom_target(lw-node-wasm-package
        DEPENDS "${LW_PPOCR_C_NODE_PACKAGE_ARCHIVE}"
                "${LW_PPOCR_C_NODE_PACKAGE_CHECKSUM}")
    endif()
endif()

if(EMSCRIPTEN AND LW_WASM_SIMD128)
    add_executable(wasm-erf-contract tests/wasm_erf_contract_driver.c)
    target_include_directories(wasm-erf-contract PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
    target_compile_features(wasm-erf-contract PRIVATE c_std_11)
    target_compile_options(wasm-erf-contract PRIVATE
        -O3 -Wall -Wextra -Wpedantic -Werror -msimd128)
    target_link_options(wasm-erf-contract PRIVATE -O3 -msimd128 "-sENVIRONMENT=node")
endif()

if(LW_WASM_COMPILED_REC_ENABLED)
    add_executable(wasm-pointwise-contract tests/wasm_pointwise_contract_driver.c)
    target_include_directories(wasm-pointwise-contract PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simd"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
    target_compile_features(wasm-pointwise-contract PRIVATE c_std_11)
    target_compile_options(wasm-pointwise-contract PRIVATE
        -O3 -Wall -Wextra -Wpedantic -Werror -msimd128)
    target_link_libraries(wasm-pointwise-contract PRIVATE lw_ppocr_c)
    target_link_options(wasm-pointwise-contract PRIVATE -O3 -msimd128 "-sENVIRONMENT=node")

    # Keep the bitwise generic/prepared kernel contract even though the
    # compiled prepared-offset experiment did not meet the E2E threshold.
    add_executable(wasm-dense-prepared-contract tests/wasm_dense_prepared_contract_driver.c)
    target_include_directories(wasm-dense-prepared-contract PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
    target_compile_features(wasm-dense-prepared-contract PRIVATE c_std_11)
    target_compile_options(wasm-dense-prepared-contract PRIVATE
        -O3 -Wall -Wextra -Wpedantic -Werror -msimd128)
    target_link_libraries(wasm-dense-prepared-contract PRIVATE lw_ppocr_c)
    target_link_options(wasm-dense-prepared-contract PRIVATE -O3 -msimd128 "-sENVIRONMENT=node")
endif()

if(EMSCRIPTEN AND LW_WASM_REC_LAZY_FALLBACK)
    # Private CI executable: exercise a source whose canonical REC session was
    # released, a compiled clone, and a clone forced onto canonical fallback.
    add_executable(wasm-rec-lazy-contract tests/wasm_rec_lazy_contract_driver.c)
    target_link_libraries(wasm-rec-lazy-contract PRIVATE lw_ppocr_c m)
    target_include_directories(wasm-rec-lazy-contract PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ppocr")
    target_compile_features(wasm-rec-lazy-contract PRIVATE c_std_11)
    target_compile_definitions(wasm-rec-lazy-contract PRIVATE
        LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
    if(LW_WASM_FINE_REC_WIDTHS)
        target_compile_definitions(wasm-rec-lazy-contract PRIVATE LW_WASM_FINE_REC_WIDTHS=1)
    endif()
    target_compile_options(wasm-rec-lazy-contract PRIVATE
        -O3 -Wall -Wextra -Wpedantic -Werror -msimd128)
    target_link_options(wasm-rec-lazy-contract PRIVATE
        -O3 -msimd128 "-sENVIRONMENT=node" "-sFORCE_FILESYSTEM=1"
        "-sALLOW_MEMORY_GROWTH=1"
        "SHELL:--embed-file \"${LW_PPOCR_C_REC_LWM}@/models/rec.lwm\""
        "SHELL:--embed-file \"${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt@/models/ppocr_keys.txt\"")
    add_dependencies(wasm-rec-lazy-contract convert_rec_model)
endif()
