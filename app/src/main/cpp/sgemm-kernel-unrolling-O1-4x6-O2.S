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
#define UNROLL_BLK_SIZE #4
#define R_TMP r7

#define R_B_ROW_1 q1
#define R_B_ROW_2 q2
#define R_B_ROW_3 q3
#define R_B_ROW_4 q4
//q15
//q0

#define R_A_COL_1 q5
#define R_A_COL_2 q6
#define R_A_COL_3 q7
#define R_A_COL_4 q8

#define R_C_COL_1 q5
#define R_C_COL_2 q6
#define R_C_COL_3 q7
#define R_C_COL_4 q8
#define R_C_COL_5 q4
#define R_C_COL_6 q3

#define R_AB_COL_1 q9
#define R_AB_COL_2 q10
#define R_AB_COL_3 q11
#define R_AB_COL_4 q12
#define R_AB_COL_5 q13
#define R_AB_COL_6 q14

        .arm
        .text
        .align  5
        .global sgemm_micro_kernel_neon_unrolling_4x6_O2 

sgemm_micro_kernel_neon_unrolling_4x6_O2:
        push {r4 - r11}
        vpush {q1 - q8}
        vpush {q9 - q15}
        add fp, sp, #272

        vpush {q0}

        cmp R_KC, #0
        beq sgemm_kernel_end 

        vmov.f32 R_AB_COL_1, #0.0
        vmov.f32 R_AB_COL_2, #0.0
        vmov.f32 R_AB_COL_3, #0.0
        vmov.f32 R_AB_COL_4, #0.0
        vmov.f32 R_AB_COL_5, #0.0
        vmov.f32 R_AB_COL_6, #0.0
        
        and R_TMP, R_KC, #3
        asrs R_KC, R_KC, #2
        beq compute_ab_residul_loop 

compute_ab_main_loop:
        vld1.f32 {d0, d1}, [R_B:128]!
        vld1.f32 {R_A_COL_1}, [R_A:128]!

        vmla.f32 R_AB_COL_1, R_A_COL_1, d0[0]
        vmla.f32 R_AB_COL_2, R_A_COL_1, d0[1]
        vld1.f32 {d2, d3}, [R_B:128]!
        vmla.f32 R_AB_COL_3, R_A_COL_1, d1[0]
        vmla.f32 R_AB_COL_4, R_A_COL_1, d1[1]
        vld1.f32 {R_A_COL_2}, [R_A:128]!
        vmla.f32 R_AB_COL_5, R_A_COL_1, d2[0]
        vmla.f32 R_AB_COL_6, R_A_COL_1, d2[1]

        vld1.f32 {d4, d5}, [R_B:128]!

        vmla.f32 R_AB_COL_1, R_A_COL_2, d3[0]
        vmla.f32 R_AB_COL_2, R_A_COL_2, d3[1]
        vld1.f32 {d6, d7}, [R_B:128]!
        vmla.f32 R_AB_COL_3, R_A_COL_2, d4[0]
        vmla.f32 R_AB_COL_4, R_A_COL_2, d4[1]
        vld1.f32 {R_A_COL_3}, [R_A:128]!
        vmla.f32 R_AB_COL_5, R_A_COL_2, d5[0]
        vmla.f32 R_AB_COL_6, R_A_COL_2, d5[1]
        
        vld1.f32 {d8, d9}, [R_B:128]!
        vmla.f32 R_AB_COL_1, R_A_COL_3, d6[0]
        vmla.f32 R_AB_COL_2, R_A_COL_3, d6[1]
        vld1.f32 {R_A_COL_4}, [R_A:128]!
        vmla.f32 R_AB_COL_3, R_A_COL_3, d7[0]
        vmla.f32 R_AB_COL_4, R_A_COL_3, d7[1]
        vld1.f32 {d10, d11}, [R_B:128]!
        vmla.f32 R_AB_COL_5, R_A_COL_3, d8[0]
        vmla.f32 R_AB_COL_6, R_A_COL_3, d8[1]
        
        vmla.f32 R_AB_COL_1, R_A_COL_4, d9[0]
        vmla.f32 R_AB_COL_2, R_A_COL_4, d9[1]
        subs R_KC, R_KC, #1
        vmla.f32 R_AB_COL_3, R_A_COL_4, d10[0]
        vmla.f32 R_AB_COL_4, R_A_COL_4, d10[1]
        vmla.f32 R_AB_COL_5, R_A_COL_4, d11[0]
        vmla.f32 R_AB_COL_6, R_A_COL_4, d11[1]
        
        bne compute_ab_main_loop 

        cmp R_TMP, #0
        beq update_c 

compute_ab_residul_loop:
        vld1.f32 {d0, d1}, [R_B]!
        vld1.f32 {R_A_COL_1}, [R_A]!

        vmla.f32 R_AB_COL_1, R_A_COL_1, d0[0]
        vmla.f32 R_AB_COL_2, R_A_COL_1, d0[1]
        vld1.f32 {d2}, [R_B]!
        vmla.f32 R_AB_COL_3, R_A_COL_1, d1[0]
        vmla.f32 R_AB_COL_4, R_A_COL_1, d1[1]
        subs R_TMP, R_TMP, #1
        vmla.f32 R_AB_COL_5, R_A_COL_1, d2[0]
        vmla.f32 R_AB_COL_6, R_A_COL_1, d2[1]

        bne compute_ab_residul_loop 

update_c:        
        vpop {q0}

        mov R_TMP, #4
        ldr R_INC_ROW_C, INC_ROW_C
        mul R_INC_ROW_C, R_INC_ROW_C, R_TMP 
        ldr R_INC_COL_C, INC_COL_C 
        mul R_INC_COL_C, R_INC_COL_C, R_TMP 
        
        mov R_TMP, R_C 
        vld1.f32 {R_C_COL_1}, [R_TMP], R_INC_COL_C  
        vld1.f32 {R_C_COL_2}, [R_TMP], R_INC_COL_C 
        vld1.f32 {R_C_COL_3}, [R_TMP], R_INC_COL_C 
        vld1.f32 {R_C_COL_4}, [R_TMP], R_INC_COL_C 
        vld1.f32 {R_C_COL_5}, [R_TMP], R_INC_COL_C 
        vld1.f32 {R_C_COL_6}, [R_TMP] 
        
        vcmp.f32 R_BETA, #0.0
        vmrs APSR_nzcv, FPSCR
        beq beta_is_zero

        vmov s2, #1.0
        vcmp.f32 R_BETA, s2 
        vmrs APSR_nzcv, FPSCR
        beq update_c_add_alpha_mul_ab 

        vmul.f32 R_C_COL_1, R_C_COL_1, d0[1] //s1<-->d0[1]<-->beta
        vmul.f32 R_C_COL_2, R_C_COL_2, d0[1] 
        vmul.f32 R_C_COL_3, R_C_COL_3, d0[1]
        vmul.f32 R_C_COL_4, R_C_COL_4, d0[1] 
        vmul.f32 R_C_COL_5, R_C_COL_5, d0[1] 
        vmul.f32 R_C_COL_6, R_C_COL_6, d0[1] 
        b update_c_add_alpha_mul_ab 

beta_is_zero:
        vmov.f32 R_C_COL_1, #0.0
        vmov.f32 R_C_COL_2, #0.0
        vmov.f32 R_C_COL_3, #0.0
        vmov.f32 R_C_COL_4, #0.0
        vmov.f32 R_C_COL_5, #0.0
        vmov.f32 R_C_COL_6, #0.0

update_c_add_alpha_mul_ab:
        vcmp.f32 R_ALPHA, s2 
        vmrs APSR_nzcv, FPSCR
        beq update_c_add_alpha_mul_ab_alpha_1

        vmul.f32 R_AB_COL_1, R_AB_COL_1, d0[0] //alpha<-->s0<-->d0[0]
        vmul.f32 R_AB_COL_2, R_AB_COL_2, d0[0]
        vmul.f32 R_AB_COL_3, R_AB_COL_3, d0[0] 
        vmul.f32 R_AB_COL_4, R_AB_COL_4, d0[0] 
        vmul.f32 R_AB_COL_5, R_AB_COL_5, d0[0] 
        vmul.f32 R_AB_COL_6, R_AB_COL_6, d0[0] 

update_c_add_alpha_mul_ab_alpha_1:
        vadd.f32 R_C_COL_1, R_C_COL_1, R_AB_COL_1 
        vadd.f32 R_C_COL_2, R_C_COL_2, R_AB_COL_2 
        vadd.f32 R_C_COL_3, R_C_COL_3, R_AB_COL_3 
        vadd.f32 R_C_COL_4, R_C_COL_4, R_AB_COL_4 
        vadd.f32 R_C_COL_5, R_C_COL_5, R_AB_COL_5 
        vadd.f32 R_C_COL_6, R_C_COL_6, R_AB_COL_6 

store_c:
        vst1.f32 {R_C_COL_1}, [R_C], R_INC_COL_C 
        vst1.f32 {R_C_COL_2}, [R_C], R_INC_COL_C 
        vst1.f32 {R_C_COL_3}, [R_C], R_INC_COL_C 
        vst1.f32 {R_C_COL_4}, [R_C], R_INC_COL_C 
        vst1.f32 {R_C_COL_5}, [R_C], R_INC_COL_C 
        vst1.f32 {R_C_COL_6}, [R_C] 
sgemm_kernel_end:
        vpop {q9 - q15}
        vpop {q1 - q8}
        pop {r4 - r11}
        bx lr

