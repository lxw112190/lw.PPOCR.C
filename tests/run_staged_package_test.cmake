if(NOT DEFINED LW_BUILD_DIR OR NOT DEFINED LW_STAGE_DIR OR
   NOT DEFINED LW_GENERATOR OR NOT DEFINED LW_MAKE_PROGRAM OR
   NOT DEFINED LW_C_COMPILER OR NOT DEFINED LW_PYTHON OR
   NOT DEFINED LW_TEST_SCRIPT OR NOT DEFINED LW_HTTP_TEST_SCRIPT)
    message(FATAL_ERROR "staged package test arguments are required")
endif()

file(REMOVE_RECURSE "${LW_STAGE_DIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${LW_BUILD_DIR}"
            --prefix "${LW_STAGE_DIR}" --config "${LW_CONFIG}"
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "install failed:\n${install_output}\n${install_error}")
endif()

set(consumer_build "${LW_BUILD_DIR}/stage-consumer-test")
file(REMOVE_RECURSE "${consumer_build}")
set(consumer_tool_args
    -DCMAKE_MAKE_PROGRAM=${LW_MAKE_PROGRAM}
    -DCMAKE_C_COMPILER=${LW_C_COMPILER})
if(DEFINED LW_RC_COMPILER AND NOT LW_RC_COMPILER STREQUAL "")
    list(APPEND consumer_tool_args -DCMAKE_RC_COMPILER=${LW_RC_COMPILER})
endif()
if(DEFINED LW_MT AND NOT LW_MT STREQUAL "")
    list(APPEND consumer_tool_args -DCMAKE_MT=${LW_MT})
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${LW_STAGE_DIR}/examples"
            -B "${consumer_build}"
            -G "${LW_GENERATOR}"
            ${consumer_tool_args}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_PREFIX_PATH=${LW_STAGE_DIR}
    RESULT_VARIABLE configure_consumer_result
    OUTPUT_VARIABLE configure_consumer_output
    ERROR_VARIABLE configure_consumer_error)
if(NOT configure_consumer_result EQUAL 0)
    message(FATAL_ERROR
        "installed consumer configure failed:\n${configure_consumer_output}\n${configure_consumer_error}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}" --config Release
    RESULT_VARIABLE build_consumer_result
    OUTPUT_VARIABLE build_consumer_output
    ERROR_VARIABLE build_consumer_error)
if(NOT build_consumer_result EQUAL 0)
    message(FATAL_ERROR
        "installed consumer build failed:\n${build_consumer_output}\n${build_consumer_error}")
endif()

execute_process(
    COMMAND "${LW_PYTHON}" "${LW_TEST_SCRIPT}" --root "${LW_STAGE_DIR}"
            --http-script "${LW_HTTP_TEST_SCRIPT}"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_output
    ERROR_VARIABLE test_error)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "staged package smoke failed:\n${test_output}\n${test_error}")
endif()
message(STATUS "${test_output}")
