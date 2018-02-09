#ifndef NEON_SGEMM_H
#define NEON_SGEMM_H

#include <stdbool.h>
#include "include/common.h"

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
        int ldc
);

#define FUNC_DEF(_level) \
void sgemm##_level(\
        bool transa,\
        bool transb,\
        int m,\
        int n,\
        int k,\
        float alpha,\
        float* a,\
        int lda,\
        float* b,\
        int ldb,\
        float beta,\
        float* c,\
        int ldc\
);\

#ifdef USE_LEVEL_O1
    FUNC_DEF(O1)
#elif defined(USE_LEVEL_O3)
    FUNC_DEF(O3)
#elif defined(USE_OMP)
    FUNC_DEF(OMP)
#endif 

#endif //NEON_SGEMM_H
