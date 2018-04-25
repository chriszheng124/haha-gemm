#ifndef HAHA_GEMM_GEMM_H
#define HAHA_GEMM_GEMM_H

#include "../base/Contsnts.h"
#include "CLEngine.h"

HAHA_GPU_BEGIN

class GEMM : public CLEngine{
public:
    bool Calc(
            bool transa,
            bool transb,
            int m,
            int n,
            int k,
            float alpha,
            float* a,
            int lda,
            float* b,
            int ldb,
            float beta,
            float* c,
            int ldc
    );

private:
    void WaitForComplete(cl_event event);
};

HAHA_GPU_END

#endif //HAHA_GEMM_GEMM_H
