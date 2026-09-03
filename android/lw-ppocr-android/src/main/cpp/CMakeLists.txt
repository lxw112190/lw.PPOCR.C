cmake_minimum_required(VERSION 3.20)
project(lw_ppocr_android LANGUAGES C)

set(LW_BUILD_HTTP_DEMO OFF CACHE BOOL "" FORCE)
set(LW_BUILD_CSHARP_DEMOS OFF CACHE BOOL "" FORCE)
set(LW_RUNTIME_ONLY ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

set(LW_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../..")
add_subdirectory("${LW_REPO_ROOT}" "${CMAKE_BINARY_DIR}/lw-ppocr-core")

add_library(lw_ppocr_android SHARED lw_android_jni.c)
target_include_directories(lw_ppocr_android PRIVATE
    "${LW_REPO_ROOT}/include")
target_compile_features(lw_ppocr_android PRIVATE c_std_11)
target_compile_options(lw_ppocr_android PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(lw_ppocr_android PRIVATE lw_ppocr_c log android jnigraphics)
set_target_properties(lw_ppocr_android PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
