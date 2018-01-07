#ifndef NEON_SGEMM_H
#define NEON_SGEMM_H

#include <stdbool.h>

#define USE_BLIS
// #define USE_LEVEL_O1
// #define USE_LEVEL_O2
#define USE_LEVEL_O3


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
#elif defined(USE_LEVEL_O2) 
    FUNC_DEF(O2)
#elif defined(USE_LEVEL_O3)
    FUNC_DEF(O3)
#endif 

#endif //NEON_SGEMM_H
