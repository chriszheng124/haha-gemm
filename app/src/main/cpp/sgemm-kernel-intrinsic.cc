#include <arm_neon.h>

#include "blksize.h"


extern "C" void sgemm_micro_kernel_intrinsic(
        int kc,
        float alpha,
        const float* a,
        const float* b,
        float beta,
        float* c,
        int inc_row_c,
        int inc_col_c){
    int i, j, l;
    
    //
    //  Compute AB = A*B
    //

    float32x4_t ab_col_1 = vmovq_n_f32(0.0);
    float32x4_t ab_col_2 = vmovq_n_f32(0.0);
    float32x4_t ab_col_3 = vmovq_n_f32(0.0);
    float32x4_t ab_col_4 = vmovq_n_f32(0.0);

    int main_loop_num = kc/4;
    int residul_loop_num = kc%4;
    
    for(l = 0; l < main_loop_num; ++l){
        float32x4_t a_col_1 = vld1q_f32(a);
        a += 4;
        float32x4_t a_col_2 = vld1q_f32(a);
        a += 4;
        float32x4_t a_col_3 = vld1q_f32(a);
        a += 4;
        float32x4_t a_col_4 = vld1q_f32(a);
        a += 4;

        float32x4_t b_row_1 = vld1q_f32(b);
        b += 4;
        float32x4_t b_row_2 = vld1q_f32(b);
        b += 4;
        float32x4_t b_row_3 = vld1q_f32(b);
        b += 4;
        float32x4_t b_row_4 = vld1q_f32(b);
        b += 4;

        ab_col_1 = vmlaq_n_f32(ab_col_1, a_col_1, b_row_1[0]);
        ab_col_2 = vmlaq_n_f32(ab_col_2, a_col_1, b_row_1[1]);
        ab_col_3 = vmlaq_n_f32(ab_col_3, a_col_1, b_row_1[2]);
        ab_col_4 = vmlaq_n_f32(ab_col_4, a_col_1, b_row_1[3]);

        ab_col_1 = vmlaq_n_f32(ab_col_1, a_col_2, b_row_2[0]);
        ab_col_2 = vmlaq_n_f32(ab_col_2, a_col_2, b_row_2[1]);
        ab_col_3 = vmlaq_n_f32(ab_col_3, a_col_2, b_row_2[2]);
        ab_col_4 = vmlaq_n_f32(ab_col_4, a_col_2, b_row_2[3]);

        ab_col_1 = vmlaq_n_f32(ab_col_1, a_col_3, b_row_3[0]);
        ab_col_2 = vmlaq_n_f32(ab_col_2, a_col_3, b_row_3[1]);
        ab_col_3 = vmlaq_n_f32(ab_col_3, a_col_3, b_row_3[2]);
        ab_col_4 = vmlaq_n_f32(ab_col_4, a_col_3, b_row_3[3]);

        ab_col_1 = vmlaq_n_f32(ab_col_1, a_col_4, b_row_4[0]);
        ab_col_2 = vmlaq_n_f32(ab_col_2, a_col_4, b_row_4[1]);
        ab_col_3 = vmlaq_n_f32(ab_col_3, a_col_4, b_row_4[2]);
        ab_col_4 = vmlaq_n_f32(ab_col_4, a_col_4, b_row_4[3]);
    }

    for(int i = 0; i < residul_loop_num; ++i){
        float32x4_t a_col_1 = vld1q_f32(a);
        float32x4_t b_row_1 = vld1q_f32(b);
        
        ab_col_1 = vmlaq_n_f32(ab_col_1, a_col_1, b_row_1[0]);
        ab_col_2 = vmlaq_n_f32(ab_col_2, a_col_1, b_row_1[1]);
        ab_col_3 = vmlaq_n_f32(ab_col_3, a_col_1, b_row_1[2]);
        ab_col_4 = vmlaq_n_f32(ab_col_4, a_col_1, b_row_1[3]);

        a += MR;
        b += NR;
    }

    //
    //  Update C <- beta*C
    //
    float* c_tmp = c;
    float32x4_t c_col_1 = vld1q_f32(c_tmp);
    c_tmp += inc_col_c;
    float32x4_t c_col_2 = vld1q_f32(c_tmp);
    c_tmp += inc_col_c;
    float32x4_t c_col_3 = vld1q_f32(c_tmp);
    c_tmp += inc_col_c;
    float32x4_t c_col_4 = vld1q_f32(c_tmp);

    if (beta == 0.0) {
        c_col_1 = vmovq_n_f32(0.0);
        c_col_2 = vmovq_n_f32(0.0);
        c_col_3 = vmovq_n_f32(0.0);
        c_col_4 = vmovq_n_f32(0.0);
    } else if (beta != 1.0) {
        c_col_1 = vmulq_n_f32(c_col_1, beta);
        c_col_2 = vmulq_n_f32(c_col_2, beta);
        c_col_3 = vmulq_n_f32(c_col_3, beta);
        c_col_4 = vmulq_n_f32(c_col_4, beta);
    }

    //
    //  Update C <- C + alpha*AB  
    //
    if (alpha == 1.0) {
        c_col_1 = vaddq_f32(c_col_1, ab_col_1);
        c_col_2 = vaddq_f32(c_col_2, ab_col_2);
        c_col_3 = vaddq_f32(c_col_3, ab_col_3);
        c_col_4 = vaddq_f32(c_col_4, ab_col_4);
    } else {
        c_col_1 = vmlaq_n_f32(c_col_1, ab_col_1, alpha);
        c_col_2 = vmlaq_n_f32(c_col_2, ab_col_2, alpha);
        c_col_3 = vmlaq_n_f32(c_col_3, ab_col_3, alpha);
        c_col_4 = vmlaq_n_f32(c_col_4, ab_col_4, alpha);
    }

    vst1q_f32(c, c_col_1);
    c += inc_col_c;
    vst1q_f32(c, c_col_2);
    c += inc_col_c;
    vst1q_f32(c, c_col_3);
    c += inc_col_c;
    vst1q_f32(c, c_col_4);
}
