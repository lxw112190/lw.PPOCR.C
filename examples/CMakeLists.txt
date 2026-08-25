cmake_minimum_required(VERSION 3.20)

project(lw_ppocr_c_package_example LANGUAGES C)

find_package(lw.PPOCR.C CONFIG REQUIRED)

function(lw_configure_example target)
    target_link_libraries(${target} PRIVATE lw.PPOCR.C::shared)
    target_compile_features(${target} PRIVATE c_std_11)
    if(WIN32)
        target_link_libraries(${target} PRIVATE shell32)
    elseif(APPLE)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "@loader_path/../lib")
    else()
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "$ORIGIN/../lib")
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

add_executable(lw-recognize-ppm-consumer recognize_ppm.c)
lw_configure_example(lw-recognize-ppm-consumer)

add_executable(lw-classify-ppm-consumer classify_ppm.c)
lw_configure_example(lw-classify-ppm-consumer)

add_executable(lw-rec-benchmark-consumer benchmark_ppm.c)
lw_configure_example(lw-rec-benchmark-consumer)
if(WIN32)
    target_link_libraries(lw-rec-benchmark-consumer PRIVATE psapi)
endif()
