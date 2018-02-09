#define MC r0
#define KC r1
#define A r2
#define INC_ROW r3
#define INC_COL [fp, #0]
#define BUFFER [fp, #4]

#define MP r4
#define _MR r5
#define R_TMP r6


        .arm
        .text
        .align  4
        .global sgemm_pack_a

sgemm_pack_a:
        push {r4 - r11}
        vpush {q1 - q8}
        vpush {q9 - q15}
        add fp, sp, #272

        cmp KC, #0
        beq sgemm_pack_a_end 

        movs MR, MC, asr #2
        and _MR, MC, #3
        mov R_TMP, #0 
        beq pack_a_main_loop

pack_a_main_loop:
       vld1.32 {q0}, A 
       add R_TMP, A, #

sgemm_pack_a_end:
        vpop {q9 - q15}
        vpop {q1 - q8}
        pop {r4 - r11}
        bx lr
