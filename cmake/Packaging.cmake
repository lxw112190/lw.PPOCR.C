# Install layout, exported CMake package, installed-package smoke test, and CPack.

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
    examples/abi_v1_client.c
    examples/ppm_image.h
    examples/recognize_ppm.c
    DESTINATION examples)
install(DIRECTORY
    examples/csharp-common
    examples/csharp-winforms
    examples/http-server
    examples/java-jni
    DESTINATION examples
    PATTERN "bin" EXCLUDE
    PATTERN "obj" EXCLUDE)
install(FILES
    "${LW_PPOCR_C_REC_LWM}"
    "${LW_PPOCR_C_CLS_LWM}"
    "${LW_PPOCR_C_DET_LWM}"
    "${LW_PPOCR_C_REC_ONNX}" "${LW_PPOCR_C_CLS_ONNX}" "${LW_PPOCR_C_DET_ONNX}"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/ppocr_keys.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6-tiny/model.json"
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
    abi/c-abi-v1-candidate.json
    abi/runtime-contract-v1.json
    abi/c-abi-v1-layout.json
    abi/lwm-v0.1-layout.json
    abi/web-abi-v1-candidate.json
    abi/exports-v1-candidate.txt
    DESTINATION docs/abi)
install(FILES
    README.md
    README.zh-CN.md
    LICENSE
    THIRD-PARTY-NOTICES.md
    dependencies.lock.json
    sbom.cdx.json
    DESTINATION .)
install(FILES
    ci/stable-release-scope.json
    DESTINATION ci)
if(LW_BUILD_ENVIRONMENT_FILE)
    if(NOT EXISTS "${LW_BUILD_ENVIRONMENT_FILE}")
        message(FATAL_ERROR
            "LW_BUILD_ENVIRONMENT_FILE does not exist: ${LW_BUILD_ENVIRONMENT_FILE}")
    endif()
    install(FILES "${LW_BUILD_ENVIRONMENT_FILE}"
        DESTINATION .
        RENAME BUILD-ENVIRONMENT.txt)
endif()
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
                -DLW_HTTP_TEST_REQUEST_TIMEOUT=${LW_HTTP_TEST_REQUEST_TIMEOUT}
                -DLW_HTTP_TEST_PROCESS_TIMEOUT=${LW_HTTP_TEST_PROCESS_TIMEOUT}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_staged_package_test.cmake")
    set_tests_properties(staged_package PROPERTIES
        TIMEOUT "${LW_STAGED_PACKAGE_TEST_TIMEOUT}")
endif()

if(LW_PACKAGE_PLATFORM_OVERRIDE)
    set(LW_PACKAGE_PLATFORM "${LW_PACKAGE_PLATFORM_OVERRIDE}")
    if(WIN32)
        set(CPACK_GENERATOR ZIP)
    else()
        set(CPACK_GENERATOR TGZ)
    endif()
elseif(WIN32)
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
