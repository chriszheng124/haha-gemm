cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)

enable_language(ASM)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL armv7-a)
    set(ASM_SOURCE src/main/cpp/sgemm-kernel.s)
endif()

link_directories(./libs)
aux_source_directory(./src/main/cpp src)

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             # Associated headers in the same location as their source
             # file are automatically included.
             ${ASM_SOURCE}
             ${src}
             )

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

target_link_libraries( # Specifies the target library.
                       native-lib

                       # Links the target library to the log library
                       # included in the NDK.
                       libblis.a
                       libm_hard.a
                       ${log-lib} )


#if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfloat-abi=hard -mhard-float -D_NDK_MATH_NO_SOFTFP=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-warn-mismatch")

