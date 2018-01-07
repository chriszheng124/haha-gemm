
set(NDK_SYSROOT_PATH /Users/zhengzhihui/Library/Android/sdk/ndk-bundle/sysroot)
set(ANDROID_TOOLCHAIN_PATH /Users/zhengzhihui/Library/Android/sdk/ndk-bundle/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/bin)
set(ANDROID_NDK_TOOLCHAIN_CROSS_PREFIX arm-linux-androideabi)

set(CMAKE_C_COMPILER ${ANDROID_TOOLCHAIN_PATH}/${ANDROID_NDK_TOOLCHAIN_CROSS_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${ANDROID_TOOLCHAIN_PATH}/${ANDROID_NDK_TOOLCHAIN_CROSS_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${ANDROID_TOOLCHAIN_PATH}/${ANDROID_NDK_TOOLCHAIN_CROSS_PREFIX}-as)

find_program(CMAKE_AR NAMES "${ANDROID_TOOLCHAIN_PATH}/${ANDROID_NDK_TOOLCHAIN_CROSS_PREFIX}-ar")
find_program(CMAKE_RANLIB NAMES "${ANDROID_TOOLCHAIN_PATH}/${ANDROID_NDK_TOOLCHAIN_CROSS_PREFIX}-ranlib")

# Skip the platform compiler checks for cross compiling
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_ASM_COMPILER_WORKS TRUE)
mark_as_advanced(CMAKE_AR)
mark_as_advanced(CMAKE_RANLIB)

set(CMAKE_SYSTEM_NAME Android)
#set(CMAKE_SYSTEM_PROCESSOR arm)
#project(native_lib C CXX ASM)
set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
