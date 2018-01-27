#include "blksize.h"

extern "C" void sgemm_micro_kernel(
        int kc,
        float alpha,
        const float* a,
        const float* b,
        float beta,
        float* c,
        int inc_row_c,
        int inc_col_c){
    
    float ab[MR*NR] __attribute__((aligned(64)));
    int i, j, l;
    
    //
    //  Compute AB = A*B
    //
    for (l = 0; l < MR*NR; ++l) {
        ab[l] = 0;
    }
    for (l = 0; l < kc; ++l) {
        for (j = 0; j < NR; ++j) {
            for (i = 0; i < MR; ++i) {
                ab[i+j*MR] += a[i]*b[j];
            }
        }
        a += MR;
        b += NR;
    }

    //
    //  Update C <- beta*C
    //
    if (beta == 0.0) {
        for (j = 0; j < NR; ++j) {
            for (i = 0; i < MR; ++i) {
                c[i*inc_row_c+j*inc_col_c] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        for (j = 0; j < NR; ++j) {
            for (i = 0; i < MR; ++i) {
                c[i*inc_row_c+j*inc_col_c] *= beta;
            }
        }
    }

    //
    //  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
    //                                  the above layer )
    //
    if (alpha == 1.0) {
        for (j = 0; j < NR; ++j) {
            for (i = 0; i < MR; ++i) {
                c[i*inc_row_c+j*inc_col_c] += ab[i+j*MR];
            }
        }
    } else {
        for (j = 0; j < NR; ++j) {
            for (i = 0; i < MR; ++i) {
                c[i*inc_row_c+j*inc_col_c] += alpha*ab[i+j*MR];
            }
        }
    }
}
