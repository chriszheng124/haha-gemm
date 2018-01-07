        .text
        .syntax   unified

        .balign   4
        .global   asm_abs
        .thumb
        .thumb_func

asm_abs:
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @
        @ arm_result_t ne10_abs_float(arm_float_t * dst,
        @                 arm_float_t * src,
        @                 unsigned int count)
        @
        @  r0: *dst
        @  r1: *src
        @  r2: int count
        @
        @  r2: loop counter
        @
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        cbz     r2, .LoopEndFloat
        mov     r3, #0
        vmov    s2, r3

.LoopBeginFloat:
        vldr      s1, [r1]                @ Load s1 = src[i]
        add       r1, r1, #4              @ move to the next item
        vabs.f32  s1, s1                  @ get the absolute value; s1 = abs(s1 - 0)
        vstr      s1, [r0]                @ Store it back into the main memory; dst[i] = s1
        add       r0, r0, #4              @ move to the next entry
        subs      r2, r2, #1              @ count down using the current index (i--)
        bne        .LoopBeginFloat        @ Continue if  "i < count"

.LoopEndFloat:
        mov     r0, #1             @ Return NE10_OK
        bx      lr
