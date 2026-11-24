# Native x86/x64 compiled backend tests. Included by NativeTests.cmake.

    if(LW_EXPERIMENTAL_AVX2_FAST_PATH AND
       (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x64|X86)$" OR
        MSVC_C_ARCHITECTURE_ID MATCHES "^(x64|X86)$"))
        add_executable(nhwc-pointwise-benchmark-driver
            tests/nhwc_pointwise_driver.c
            src/kernels/nhwc_pack.c
            src/simd/avx2_nhwc_pointwise.c)
        target_link_libraries(nhwc-pointwise-benchmark-driver PRIVATE lw_ppocr_c)
        target_include_directories(nhwc-pointwise-benchmark-driver PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
        target_compile_features(nhwc-pointwise-benchmark-driver PRIVATE c_std_11)
        target_compile_definitions(nhwc-pointwise-benchmark-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        if(MSVC)
            set_property(SOURCE src/simd/avx2_nhwc_pointwise.c APPEND PROPERTY
                COMPILE_OPTIONS /arch:AVX2 /fp:strict)
            target_compile_definitions(nhwc-pointwise-benchmark-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(nhwc-pointwise-benchmark-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(nhwc-pointwise-benchmark-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME nhwc_pointwise_benchmark
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nhwc_pointwise_benchmark.py"
                    --driver "$<TARGET_FILE:nhwc-pointwise-benchmark-driver>")
        set_tests_properties(nhwc_pointwise_benchmark PROPERTIES TIMEOUT 120)
        add_executable(nhwc-support-driver
            tests/nhwc_support_driver.c)
        target_link_libraries(nhwc-support-driver PRIVATE lw_ppocr_c)
        target_include_directories(nhwc-support-driver PRIVATE
            src/kernels)
        target_compile_features(nhwc-support-driver PRIVATE c_std_11)
        target_compile_definitions(nhwc-support-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        if(MSVC)
            target_compile_definitions(nhwc-support-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(nhwc-support-driver PRIVATE /arch:AVX2 /W4 /WX /utf-8)
        else()
            target_compile_options(nhwc-support-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME nhwc_support_pool COMMAND nhwc-support-driver)
        set_tests_properties(nhwc_support_pool PROPERTIES TIMEOUT 30)
        add_executable(nhwc-dense-benchmark-driver
            tests/nhwc_dense_driver.c
            src/kernels/nhwc_pack.c
            src/simd/avx2_nhwc_dense.c)
        target_link_libraries(nhwc-dense-benchmark-driver PRIVATE lw_ppocr_c)
        target_include_directories(nhwc-dense-benchmark-driver PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simd")
        target_compile_features(nhwc-dense-benchmark-driver PRIVATE c_std_11)
        target_compile_definitions(nhwc-dense-benchmark-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        if(MSVC)
            set_property(SOURCE src/simd/avx2_nhwc_dense.c APPEND PROPERTY
                COMPILE_OPTIONS /arch:AVX2 /fp:strict)
            target_compile_definitions(nhwc-dense-benchmark-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(nhwc-dense-benchmark-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(nhwc-dense-benchmark-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME nhwc_dense_benchmark
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nhwc_dense_benchmark.py"
                    --driver "$<TARGET_FILE:nhwc-dense-benchmark-driver>")
        set_tests_properties(nhwc_dense_benchmark PROPERTIES TIMEOUT 120)
        add_executable(nhwc-convtranspose-driver
            tests/nhwc_convtranspose_driver.c)
        target_link_libraries(nhwc-convtranspose-driver PRIVATE lw_ppocr_c)
        target_include_directories(nhwc-convtranspose-driver PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simd"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
        target_compile_features(nhwc-convtranspose-driver PRIVATE c_std_11)
        target_compile_definitions(nhwc-convtranspose-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        if(MSVC)
            target_compile_definitions(nhwc-convtranspose-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(nhwc-convtranspose-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(nhwc-convtranspose-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME nhwc_convtranspose_contract COMMAND nhwc-convtranspose-driver)
        set_tests_properties(nhwc_convtranspose_contract PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-backend-contract-driver
            tests/x64_rec_backend_contract_driver.c)
        target_link_libraries(x64-rec-backend-contract-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-backend-contract-driver PRIVATE
            src/runtime
            src/kernels
            src/simd)
        target_compile_features(x64-rec-backend-contract-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-backend-contract-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-backend-contract-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-backend-contract-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-backend-contract-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-backend-contract-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_backend_contract
            COMMAND x64-rec-backend-contract-driver ${LW_PPOCR_C_REC_LWM})
        set_tests_properties(x64_rec_backend_contract PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-backend-execution-driver
            tests/x64_rec_backend_execution_driver.c)
        if(LW_EXPERIMENTAL_REC_DW_PW_FUSION)
            add_executable(rec-dw-pw-contract-driver tests/rec_dw_pw_contract_driver.c)
            target_link_libraries(rec-dw-pw-contract-driver PRIVATE lw_ppocr_c)
            target_include_directories(rec-dw-pw-contract-driver PRIVATE src/runtime src/kernels src/simd)
            target_compile_features(rec-dw-pw-contract-driver PRIVATE c_std_11)
            if(MSVC)
                target_compile_options(rec-dw-pw-contract-driver PRIVATE /W4 /WX /utf-8)
            else()
                target_compile_options(rec-dw-pw-contract-driver PRIVATE -Wall -Wextra -Wpedantic -Werror)
            endif()
            foreach(variant tiny small medium)
                add_test(NAME rec_dw_pw_contract_${variant}
                    COMMAND rec-dw-pw-contract-driver
                        "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-${variant}/rec.onnx")
                set_tests_properties(rec_dw_pw_contract_${variant} PROPERTIES TIMEOUT 180)
            endforeach()
        endif()
        if(LW_EXPERIMENTAL_REC_FFN_TILING OR LW_EXPERIMENTAL_REC_FFN_FUSION)
            add_executable(rec-ffn-contract-driver tests/rec_ffn_contract_driver.c)
            target_link_libraries(rec-ffn-contract-driver PRIVATE lw_ppocr_c)
            target_include_directories(rec-ffn-contract-driver PRIVATE src/runtime src/kernels src/simd)
            target_compile_features(rec-ffn-contract-driver PRIVATE c_std_11)
            if(LW_EXPERIMENTAL_REC_FFN_FUSION)
                target_compile_definitions(rec-ffn-contract-driver PRIVATE LW_EXPERIMENTAL_REC_FFN_FUSION=1)
            endif()
            if(MSVC)
                target_compile_options(rec-ffn-contract-driver PRIVATE /W4 /WX /utf-8)
            else()
                target_compile_options(rec-ffn-contract-driver PRIVATE -Wall -Wextra -Wpedantic -Werror)
            endif()
            foreach(variant tiny small medium)
                add_test(NAME rec_ffn_contract_${variant}
                    COMMAND rec-ffn-contract-driver
                        "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-${variant}/rec.onnx")
                set_tests_properties(rec_ffn_contract_${variant} PROPERTIES TIMEOUT 180)
            endforeach()
        endif()
        target_link_libraries(x64-rec-backend-execution-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-backend-execution-driver PRIVATE
            src/runtime
            src/kernels
            src/simd
            src/ppocr)
        target_compile_features(x64-rec-backend-execution-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-backend-execution-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-backend-execution-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-backend-execution-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-backend-execution-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-backend-execution-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_backend_execution
            COMMAND x64-rec-backend-execution-driver ${LW_PPOCR_C_REC_LWM})
        set_tests_properties(x64_rec_backend_execution PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-shared-constants-driver
            tests/x64_rec_shared_constants_driver.c)
        target_link_libraries(x64-rec-shared-constants-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-shared-constants-driver PRIVATE
            src/runtime src/kernels src/simd)
        target_compile_features(x64-rec-shared-constants-driver PRIVATE c_std_11)
        add_dependencies(x64-rec-shared-constants-driver convert_rec_model)
        if(MSVC)
            target_compile_options(x64-rec-shared-constants-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-shared-constants-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_shared_constants
            COMMAND x64-rec-shared-constants-driver ${LW_PPOCR_C_REC_LWM})
        set_tests_properties(x64_rec_shared_constants PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-backend-nchw-driver
            tests/x64_rec_backend_nchw_driver.c)
        target_link_libraries(x64-rec-backend-nchw-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-backend-nchw-driver PRIVATE
            src/runtime
            src/kernels
            src/simd
            src/ppocr)
        target_compile_features(x64-rec-backend-nchw-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-backend-nchw-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-backend-nchw-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-backend-nchw-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-backend-nchw-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-backend-nchw-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_backend_nchw
            COMMAND x64-rec-backend-nchw-driver ${LW_PPOCR_C_REC_LWM})
        set_tests_properties(x64_rec_backend_nchw PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-layout-benchmark-driver
            tests/x64_rec_layout_benchmark_driver.c)
        target_link_libraries(x64-rec-layout-benchmark-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-layout-benchmark-driver PRIVATE
            src/runtime
            src/kernels
            src/simd
            src/ppocr)
        target_compile_features(x64-rec-layout-benchmark-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-layout-benchmark-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-layout-benchmark-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-layout-benchmark-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-layout-benchmark-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-layout-benchmark-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_layout_benchmark
            COMMAND x64-rec-layout-benchmark-driver ${LW_PPOCR_C_REC_LWM} 5)
        set_tests_properties(x64_rec_layout_benchmark PROPERTIES TIMEOUT 120)
        foreach(rec_width IN ITEMS 192 320 480 640)
            add_test(NAME x64_rec_layout_benchmark_${rec_width}
                COMMAND x64-rec-layout-benchmark-driver ${LW_PPOCR_C_REC_LWM}
                        5 ${rec_width})
            set_tests_properties(x64_rec_layout_benchmark_${rec_width}
                PROPERTIES TIMEOUT 120)
        endforeach()
        add_executable(x64-det-backend-contract-driver
            tests/x64_det_backend_contract_driver.c)
        target_link_libraries(x64-det-backend-contract-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-det-backend-contract-driver PRIVATE
            src/runtime
            src/kernels
            src/simd
            src/ppocr)
        target_compile_features(x64-det-backend-contract-driver PRIVATE c_std_11)
        target_compile_definitions(x64-det-backend-contract-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-det-backend-contract-driver convert_det_model)
        if(MSVC)
            target_compile_definitions(x64-det-backend-contract-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-det-backend-contract-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-det-backend-contract-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_det_backend_contract
            COMMAND x64-det-backend-contract-driver ${LW_PPOCR_C_DET_LWM} 32 64)
        set_tests_properties(x64_det_backend_contract PROPERTIES TIMEOUT 120)
        add_test(NAME x64_det_backend_contract_sharded
            COMMAND x64-det-backend-contract-driver ${LW_PPOCR_C_DET_LWM} 640 640)
        set_tests_properties(x64_det_backend_contract_sharded PROPERTIES TIMEOUT 180)
        add_executable(x64-det-layout-benchmark-driver
            tests/x64_det_layout_benchmark_driver.c)
        target_link_libraries(x64-det-layout-benchmark-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-det-layout-benchmark-driver PRIVATE
            src/runtime
            src/kernels
            src/simd
            src/ppocr)
        target_compile_features(x64-det-layout-benchmark-driver PRIVATE c_std_11)
        target_compile_definitions(x64-det-layout-benchmark-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-det-layout-benchmark-driver convert_det_model)
        if(MSVC)
            target_compile_definitions(x64-det-layout-benchmark-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-det-layout-benchmark-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-det-layout-benchmark-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_det_layout_benchmark
            COMMAND x64-det-layout-benchmark-driver ${LW_PPOCR_C_DET_LWM} 32 64 1 3)
        set_tests_properties(x64_det_layout_benchmark PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-backend-benchmark-driver
            tests/x64_rec_backend_benchmark_driver.c)
        target_link_libraries(x64-rec-backend-benchmark-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-backend-benchmark-driver PRIVATE
            src/runtime
            src/kernels
            src/simd
            src/ppocr)
        target_compile_features(x64-rec-backend-benchmark-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-backend-benchmark-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-backend-benchmark-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-backend-benchmark-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-backend-benchmark-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-backend-benchmark-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_backend_benchmark
            COMMAND x64-rec-backend-benchmark-driver ${LW_PPOCR_C_REC_LWM} 5)
        set_tests_properties(x64_rec_backend_benchmark PROPERTIES TIMEOUT 120)
        add_test(NAME x64_rec_backend_benchmark_nchw
            COMMAND x64-rec-backend-benchmark-driver ${LW_PPOCR_C_REC_LWM} 5 nchw)
        set_tests_properties(x64_rec_backend_benchmark_nchw PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-recognizer-benchmark-driver
            tests/x64_rec_recognizer_benchmark_driver.c)
        target_link_libraries(x64-rec-recognizer-benchmark-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-recognizer-benchmark-driver PRIVATE
            src/ppocr)
        target_compile_features(x64-rec-recognizer-benchmark-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-recognizer-benchmark-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-recognizer-benchmark-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-recognizer-benchmark-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-recognizer-benchmark-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-recognizer-benchmark-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_recognizer_benchmark
            COMMAND x64-rec-recognizer-benchmark-driver ${LW_PPOCR_C_REC_LWM}
                ${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt 5)
        foreach(rec_width IN ITEMS 192 320 480 640)
            add_test(NAME x64_rec_recognizer_benchmark_${rec_width}
                COMMAND x64-rec-recognizer-benchmark-driver ${LW_PPOCR_C_REC_LWM}
                    ${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt
                    5 ${rec_width})
            set_tests_properties(x64_rec_recognizer_benchmark_${rec_width}
                PROPERTIES TIMEOUT 120)
        endforeach()
        set_tests_properties(x64_rec_recognizer_benchmark PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-pointwise-layout-driver
            tests/x64_rec_pointwise_layout_driver.c)
        target_link_libraries(x64-rec-pointwise-layout-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-pointwise-layout-driver PRIVATE
            src/runtime
            src/kernels
            src/simd)
        target_compile_features(x64-rec-pointwise-layout-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-pointwise-layout-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-pointwise-layout-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-pointwise-layout-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-pointwise-layout-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-pointwise-layout-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_pointwise_layout
            COMMAND x64-rec-pointwise-layout-driver ${LW_PPOCR_C_REC_LWM} 5)
        set_tests_properties(x64_rec_pointwise_layout PROPERTIES TIMEOUT 120)
        add_executable(x64-rec-pointwise-tuning-driver
            tests/x64_rec_pointwise_tuning_driver.c)
        target_link_libraries(x64-rec-pointwise-tuning-driver PRIVATE lw_ppocr_c)
        target_include_directories(x64-rec-pointwise-tuning-driver PRIVATE
            src/runtime
            src/kernels
            src/simd
            src/ppocr)
        target_compile_features(x64-rec-pointwise-tuning-driver PRIVATE c_std_11)
        target_compile_definitions(x64-rec-pointwise-tuning-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(x64-rec-pointwise-tuning-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(x64-rec-pointwise-tuning-driver PRIVATE
                _CRT_SECURE_NO_WARNINGS)
            target_compile_options(x64-rec-pointwise-tuning-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(x64-rec-pointwise-tuning-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_executable(rec-fast-prototype-driver tests/rec_fast_prototype_driver.c)
        target_link_libraries(rec-fast-prototype-driver PRIVATE lw_ppocr_c)
        target_include_directories(rec-fast-prototype-driver PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
        target_compile_features(rec-fast-prototype-driver PRIVATE c_std_11)
        target_compile_definitions(rec-fast-prototype-driver PRIVATE
            LW_EXPERIMENTAL_AVX2_FAST_PATH=1)
        add_dependencies(rec-fast-prototype-driver convert_rec_model)
        if(MSVC)
            target_compile_definitions(rec-fast-prototype-driver PRIVATE _CRT_SECURE_NO_WARNINGS)
            target_compile_options(rec-fast-prototype-driver PRIVATE /W4 /WX /utf-8)
        else()
            target_compile_options(rec-fast-prototype-driver PRIVATE
                -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME x64_rec_fast_prototype
            COMMAND rec-fast-prototype-driver "${LW_PPOCR_C_REC_LWM}" 960)
        set_tests_properties(x64_rec_fast_prototype PROPERTIES TIMEOUT 120)
    elseif(LW_EXPERIMENTAL_AVX2_FAST_PATH)
        message(WARNING
            "LW_EXPERIMENTAL_AVX2_FAST_PATH requires an x86/x64 build; benchmark target skipped")
    endif()
