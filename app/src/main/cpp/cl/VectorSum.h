#ifndef HAHA_GEMM_REDUCTION_H
#define HAHA_GEMM_REDUCTION_H


#include "CLEngine.h"

HAHA_GPU_BEGIN

class VectorSum : public CLEngine{
public:
    bool Sum(float* a, float* b, uint length, float* c);
    bool Sum2(int* a, int* b, uint length, int* c, const char* kernel_code, float* run_time);

    bool ValidResult(float* a, float*b, uint length, float* c);
    bool ValidResult(int* a, int* b, uint length, int* c);
};

HAHA_GPU_END

#endif //HAHA_GEMM_REDUCTION_H
