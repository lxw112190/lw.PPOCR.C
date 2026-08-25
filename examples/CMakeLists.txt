cmake_minimum_required(VERSION 3.20)

project(lw_ppocr_c_package_example LANGUAGES C)

find_package(lw.PPOCR.C CONFIG REQUIRED)

add_executable(lw-recognize-ppm-consumer recognize_ppm.c)
target_link_libraries(lw-recognize-ppm-consumer PRIVATE lw.PPOCR.C::shared)
target_compile_features(lw-recognize-ppm-consumer PRIVATE c_std_11)
if(WIN32)
    target_link_libraries(lw-recognize-ppm-consumer PRIVATE shell32)
elseif(APPLE)
    set_target_properties(lw-recognize-ppm-consumer PROPERTIES
        BUILD_RPATH "@loader_path/../lib")
else()
    set_target_properties(lw-recognize-ppm-consumer PROPERTIES
        BUILD_RPATH "$ORIGIN/../lib")
endif()
if(MSVC)
    target_compile_options(lw-recognize-ppm-consumer PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw-recognize-ppm-consumer PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
