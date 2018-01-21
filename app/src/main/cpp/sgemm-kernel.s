#define R_KC r0
#define R_ALPHA s0
#define R_A r1
#define R_B r2
#define R_BETA s1
#define R_C r3
#define INC_ROW_C [fp, #0]
#define INC_COL_C [fp, #4]
#define R_INC_ROW_C r4
#define R_INC_COL_C r5

#define R_L r6
#define MR #4
#define NR #4
#define R_TMP r7

#define R_AB_COL_1 q3
#define R_AB_COL_2 q4
#define R_AB_COL_3 q5
#define R_AB_COL_4 q6

        .arm
        .text
        .align  5
        .global sgemm_micro_kernel_neon 

sgemm_micro_kernel_neon:
        push {r4 - r11}
        vpush {q1 - q8}
        vpush {q9 - q15}
        add fp, sp, #272

        mov R_L, R_KC  
        cmp R_L, #0
        beq sgemm_kernel_end 

        vmov.f32 R_AB_COL_1, #0.0
        vmov.f32 R_AB_COL_2, #0.0
        vmov.f32 R_AB_COL_3, #0.0
        vmov.f32 R_AB_COL_4, #0.0
compute_ab:
        vld1.f32 {q1}, [R_A]!
        vld1.f32 {q2}, [R_B]!

        vmla.f32 R_AB_COL_1, q1, d4[0]
        vmla.f32 R_AB_COL_2, q1, d4[1]
        vmla.f32 R_AB_COL_3, q1, d5[0]
        vmla.f32 R_AB_COL_4, q1, d5[1]

        subs R_L, R_L, #1
        bne compute_ab 

        mov R_TMP, #4
        ldr R_INC_ROW_C, INC_ROW_C
        mul R_INC_ROW_C, R_INC_ROW_C, R_TMP 
        ldr R_INC_COL_C, INC_COL_C 
        mul R_INC_COL_C, R_INC_COL_C, R_TMP 
        
        mov R_TMP, R_C 
        vld1.f32 {q7}, [R_TMP], R_INC_COL_C  
        vld1.f32 {q8}, [R_TMP], R_INC_COL_C 
        vld1.f32 {q9}, [R_TMP], R_INC_COL_C 
        vld1.f32 {q10}, [R_TMP]  
        
        vcmp.f32 R_BETA, #0.0
        vmrs APSR_nzcv, FPSCR
        beq beta_is_zero

        vmov s2, #1.0
        vcmp.f32 R_BETA, s2 
        beq update_c_add_alpha_mul_ab 

        vmul.f32 q7, q7, d0[1] //s1<-->d0[1]<-->beta
        vmul.f32 q8, q8, d0[1] 
        vmul.f32 q9, q9, d0[1]
        vmul.f32 q10, q10, d0[1] 
        b update_c_add_alpha_mul_ab 

beta_is_zero:
        vmov.f32 q7, #0.0
        vmov.f32 q8, #0.0
        vmov.f32 q9, #0.0
        vmov.f32 q10, #0.0

update_c_add_alpha_mul_ab:
        vcmp.f32 R_ALPHA, s2 
        vmrs APSR_nzcv, FPSCR
        beq update_c_add_alpha_mul_ab_alpha_1

        vmul.f32 R_AB_COL_1, R_AB_COL_1, d0[0] //alpha<-->s0<-->d0[0]
        vmul.f32 R_AB_COL_2, R_AB_COL_2, d0[0]
        vmul.f32 R_AB_COL_3, R_AB_COL_3, d0[0] 
        vmul.f32 R_AB_COL_4, R_AB_COL_4, d0[0] 

update_c_add_alpha_mul_ab_alpha_1:
        vadd.f32 q7, q7, R_AB_COL_1 
        vadd.f32 q8, q8, R_AB_COL_2 
        vadd.f32 q9, q9, R_AB_COL_3 
        vadd.f32 q10, q10, R_AB_COL_4 

store_c:
        vst1.f32 {q7}, [R_C], R_INC_COL_C 
        vst1.f32 {q8}, [R_C], R_INC_COL_C 
        vst1.f32 {q9}, [R_C], R_INC_COL_C 
        vst1.f32 {q10}, [R_C]
sgemm_kernel_end:
        vpop {q9 - q15}
        vpop {q1 - q8}
        pop {r4 - r11}
        bx lr

