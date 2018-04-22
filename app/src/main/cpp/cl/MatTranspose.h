#ifndef HAHA_GEMM_MATTRANSPOSE_H
#define HAHA_GEMM_MATTRANSPOSE_H


#include "CLEngine.h"

HAHA_GPU_BEGIN

class MatTranspose : public CLEngine{
public:
    MatTranspose(const char* kernel_code);

    bool Transpose(float* in, float* out);

private:
    static const char* kKernelFuncName = "transpose";
    char* kernel_code_;
};

HAHA_GPU_END

#endif //HAHA_GEMM_MATTRANSPOSE_H
