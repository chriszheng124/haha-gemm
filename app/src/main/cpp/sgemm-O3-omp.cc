#include <omp.h>
#include <iostream>

#include "sgemm.h"
#include "utils.h"
#include "blksize.h"

using namespace HahaGemm;

static float PACKED_A[MC*KC] __attribute__((aligned(64)));
static float PACKED_B[KC*NC] __attribute__((aligned(64)));

#define DECLARE_FUNC(_FUNC_NAME)\
    extern "C" void _FUNC_NAME(\
        int kc,\
        float alpha,\
        const float* a,\
        const float* b,\
        float beta,\
        float* c,\
        int inc_row_c,\
        int inc_col_c);\

DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_O3)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_4x6_O2)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_4x6_O2_pld)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_O2)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_O1)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_4x6)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_4x6_slow)
DECLARE_FUNC(sgemm_micro_kernel_neon_unrolling_4x6_Os)
DECLARE_FUNC(sgemm_micro_kernel_neon)
DECLARE_FUNC(haha_sgemm_micro_kernel)
DECLARE_FUNC(sgemm_micro_kernel_4x4_O1)
DECLARE_FUNC(sgemm_micro_kernel_4x4_O2)
DECLARE_FUNC(sgemm_micro_kernel_intrinsic)
DECLARE_FUNC(sgemm_micro_kernel_neon_4x4)
DECLARE_FUNC(sgemm_micro_kernel_neon_4x4_pld)
DECLARE_FUNC(sgemm_micro_kernel_unrolling_4x8)

static void sgemm_micro_kernel_wrapper(
        int kc,
        float alpha,
        const float* a,
        const float* b,
        float beta,
        float* c,
        int inc_row_c,
        int inc_col_c){
#ifdef USE_4X8_R_BLOCK 
    sgemm_micro_kernel_unrolling_4x8(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
#elif defined USE_4X6_R_BLOCK 
    // sgemm_micro_kernel_neon_unrolling_4x6_Os(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    sgemm_micro_kernel_intrinsic(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_4x6_slow(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_4x6(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_4x6_O2(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
#else 
    // sgemm_micro_kernel_4x4_O2(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    sgemm_micro_kernel_4x4_O1(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // haha_sgemm_micro_kernel(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_O1(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_O2(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_O3(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_4x4(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_4x4_pld(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
#endif 
}

static void haha_sgemm_macro_block(int mc,
                   int     nc,
                   int     kc,
                   float  alpha,
                   float  beta,
                   float  *c,
                   int     inc_row_c,
                   int     inc_col_c){
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

// #pragma omp parallel 
#pragma omp parallel for// nowait //num_threads(4)
    for (int j = 0; j < np; ++j) {
        int nr = (j != np-1 || _nr == 0) ? NR : _nr;
        for (int i = 0; i < mp; ++i) {
            int mr = (i != mp-1 || _mr == 0) ? MR : _mr;
            if (mr == MR && nr == NR) {
                sgemm_micro_kernel_wrapper(kc, alpha, &PACKED_A[i*kc*MR], &PACKED_B[j*kc*NR],
                                   beta, &c[i*MR*inc_row_c+j*NR*inc_col_c],
                                   inc_row_c, inc_col_c);
            } else {
                float REGISTER_BLOCK_C[MR*NR] __attribute__((aligned(64)));
                
                sgemm_micro_kernel_wrapper(kc, alpha, &PACKED_A[i*kc*MR], &PACKED_B[j*kc*NR],
                                   0.0, REGISTER_BLOCK_C, 1, MR);
                Utils::Scale(mr, nr, beta,
                        &c[i*MR*inc_row_c+j*NR*inc_col_c], inc_row_c, inc_col_c);
                Utils::ScaleAdd(mr, nr, 1.0, REGISTER_BLOCK_C, 1, MR,
                        &c[i*MR*inc_row_c+j*NR*inc_col_c], inc_row_c, inc_col_c);
            }
        }
    }
}

void sgemmOMP(
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
    int mb = (m + MC - 1) / MC;
    int nb = (n + NC - 1) / NC;
    int kb = (k + KC - 1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int nc, kc;

    float _beta;

    if (alpha == 0.0 || k == 0) {
        Utils::Scale(m, n, beta, c, 1, ldc);
        return;
    }

    for (int j = 0; j < nb; ++j) {
        nc = (j != nb-1 || _nc == 0) ? NC : _nc;

        for (int l = 0; l < kb; ++l) {
            kc    = (l != kb-1 || _kc == 0) ? KC : _kc;
            _beta = (l == 0) ? beta : 1.0f;

            Utils::PackB(kc, nc, &b[l*KC*1+j*NC*ldb], 1, ldb, PACKED_B);

            for (int i = 0; i < mb; ++i) {
                int mc = (i != mb-1 || _mc == 0) ? MC : _mc;
                
                Utils::PackA(mc, kc, &a[i*MC*1+l*KC*lda], 1, lda, PACKED_A);

                haha_sgemm_macro_block(mc, nc, kc, alpha, _beta,
                                   &c[i*MC*1+j*NC*ldc], 1, ldc);
            }
        }
    }
}

