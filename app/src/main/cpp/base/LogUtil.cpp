#include <android/log.h>
#include <stdarg.h>
#include <cstdio>

#include "LogUtil.h"


void LogUtil::V(const char *fmt, ...) {
    char buf[1024];
    android_LogPriority t = ANDROID_LOG_INFO;
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    __android_log_vprint(t, "haha_gemm", fmt, args);
}

void LogUtil::E(const char *fmt, ...) {
    char buf[1024];
    android_LogPriority t = ANDROID_LOG_ERROR;
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    __android_log_vprint(t, "haha_gemm", fmt, args);
}

