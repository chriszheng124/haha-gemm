#include <iostream>
#include "sgemm.h"
#include "utils.h"

using namespace HahaGemm;

void sgemm(
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
        int ldc){
    long current_time = Utils::GetCurrentTimeMs();
#ifdef USE_LEVEL_O1
    sgemmO1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#elif defined(USE_LEVEL_O3) 
    sgemmO3(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#elif defined(USE_OMP)
    sgemmOMP(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif

#ifndef VERIFY_RESULT 
    std::cout<<"Gemm Using time : "<<Utils::GetCurrentTimeMs() - current_time <<std::endl;
#endif 
}

