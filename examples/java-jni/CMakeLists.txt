cmake_minimum_required(VERSION 3.20)

project(lw_ppocr_java_jni_example LANGUAGES C)

find_package(JNI REQUIRED)
find_package(lw.PPOCR.C CONFIG REQUIRED)

add_library(lw_ppocr_java SHARED native/lw_java_jni.c)
target_compile_features(lw_ppocr_java PRIVATE c_std_11)
target_include_directories(lw_ppocr_java PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(lw_ppocr_java PRIVATE lw.PPOCR.C::shared)

if(MSVC)
    target_compile_options(lw_ppocr_java PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(lw_ppocr_java PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(lw_ppocr_java PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(lw_ppocr_java PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path")
endif()

add_custom_command(
    TARGET lw_ppocr_java
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:lw.PPOCR.C::shared>
            $<TARGET_FILE_DIR:lw_ppocr_java>)

if(WIN32)
    # The installed Windows package carries the matching MSVC/UCRT runtime
    # DLLs beside lw_ppocr_c.dll. Copy that runtime closure next to the JNI
    # library so the documented java.library.path command works in a clean
    # shell, not only inside a Visual Studio developer prompt.
    add_custom_command(
        TARGET lw_ppocr_java
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                $<TARGET_FILE_DIR:lw.PPOCR.C::shared>
                $<TARGET_FILE_DIR:lw_ppocr_java>)
endif()
