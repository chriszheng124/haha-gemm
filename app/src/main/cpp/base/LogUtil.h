#ifndef HAHA_GEMM_LOGUTIL_H
#define HAHA_GEMM_LOGUTIL_H


class LogUtil {
public:
    static void V(const char* fmt, ...);
    static void E(const char* fmt, ...);
};


#endif //HAHA_GEMM_LOGUTIL_H
