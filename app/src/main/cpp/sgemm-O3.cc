#include "sgemm.h"
#include "utils.h"
#include "blksize.h"

using namespace HahaGemm;

static float PACKED_A[MC*KC] __attribute__((aligned(64)));
static float PACKED_B[KC*NC] __attribute__((aligned(64)));
static float REGISTER_BLOCK_C[MR*NR] __attribute__((aligned(64)));


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
DECLARE_FUNC(sgemm_micro_kernel_neon)
DECLARE_FUNC(sgemm_micro_kernel)
DECLARE_FUNC(sgemm_micro_kernel_4x4_O1)
DECLARE_FUNC(sgemm_micro_kernel_4x4_O2)

static void sgemm_micro_kernel_wrapper(
        int kc,
        float alpha,
        const float* a,
        const float* b,
        float beta,
        float* c,
        int inc_row_c,
        int inc_col_c){
    // sgemm_micro_kernel_4x4_O2(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_4x4_O1(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_O1(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_O2(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_O3(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_4x6(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    // sgemm_micro_kernel_neon_unrolling_4x6_O2(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
    sgemm_micro_kernel_neon_unrolling_4x6_O2_pld(kc, alpha, a, b, beta, c, inc_row_c, inc_col_c);
}

static void sgemm_macro_block(int mc,
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

    int mr, nr;
    int i, j;

    for (j = 0; j < np; ++j) {
        nr = (j != np-1 || _nr == 0) ? NR : _nr;
        for (i = 0; i < mp; ++i) {
            mr = (i != mp-1 || _mr == 0) ? MR : _mr;
            if (mr == MR && nr == NR) {
                sgemm_micro_kernel_wrapper(kc, alpha, &PACKED_A[i*kc*MR], &PACKED_B[j*kc*NR],
                                   beta, &c[i*MR*inc_row_c+j*NR*inc_col_c],
                                   inc_row_c, inc_col_c);
            } else {
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

void sgemmO3(
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

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (alpha == 0.0 || k == 0) {
        Utils::Scale(m, n, beta, c, 1, ldc);
        return;
    }

    for (j = 0; j < nb; ++j) {
        nc = (j != nb-1 || _nc == 0) ? NC : _nc;

        for (l = 0; l < kb; ++l) {
            kc    = (l != kb-1 || _kc == 0) ? KC : _kc;
            _beta = (l == 0) ? beta : 1.0f;

            Utils::PackB(kc, nc, &b[l*KC*1+j*NC*ldb], 1, ldb, PACKED_B);

            for (i = 0; i < mb; ++i) {
                mc = (i != mb-1 || _mc == 0) ? MC : _mc;

                Utils::PackA(mc, kc, &a[i*MC*1+l*KC*lda], 1, lda, PACKED_A);

                sgemm_macro_block(mc, nc, kc, alpha, _beta,
                                   &c[i*MC*1+j*NC*ldc], 1, ldc);
            }
        }
    }
}

