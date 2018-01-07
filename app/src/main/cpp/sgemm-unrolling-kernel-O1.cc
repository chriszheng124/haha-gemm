#include "blksize.h"

#define SET_ZERO_4X4(_ARRAY)\
    do{\
        _ARRAY[0] = 0;\
        _ARRAY[1] = 0;\
        _ARRAY[2] = 0;\
        _ARRAY[3] = 0;\
        _ARRAY[4] = 0;\
        _ARRAY[5] = 0;\
        _ARRAY[6] = 0;\
        _ARRAY[7] = 0;\
        _ARRAY[8] = 0;\
        _ARRAY[9] = 0;\
        _ARRAY[10] = 0;\
        _ARRAY[11] = 0;\
        _ARRAY[12] = 0;\
        _ARRAY[13] = 0;\
        _ARRAY[14] = 0;\
        _ARRAY[15] = 0;\
    }while(0)

void sgemm_micro_kernel_4x4_O1(
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
    SET_ZERO_4X4(ab);
    
    for(l = 0; l < kc; ++l){
        ab[0] += a[0] * b[0];
        ab[4] += a[0] * b[1];
        ab[8] += a[0] * b[2];
        ab[12] += a[0] * b[3];
        
        ab[1] += a[1] * b[0];
        ab[5] += a[1] * b[1];
        ab[9] += a[1] * b[2];
        ab[13] += a[1] * b[3];

        ab[2] += a[2] * b[0];
        ab[6] += a[2] * b[1];
        ab[10] += a[2] * b[2];
        ab[14] += a[2] * b[3];
        
        ab[3] += a[3] * b[0];
        ab[7] += a[3] * b[1];
        ab[11] += a[3] * b[2];
        ab[15] += a[3] * b[3];
        
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
    //  Update C <- C + alpha*AB  
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
