if(NOT DEFINED LW_BUILD_DIR OR NOT DEFINED LW_STAGE_DIR OR
   NOT DEFINED LW_PYTHON OR NOT DEFINED LW_TEST_SCRIPT)
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

execute_process(
    COMMAND "${LW_PYTHON}" "${LW_TEST_SCRIPT}" --root "${LW_STAGE_DIR}"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_output
    ERROR_VARIABLE test_error)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "staged package smoke failed:\n${test_output}\n${test_error}")
endif()
message(STATUS "${test_output}")
