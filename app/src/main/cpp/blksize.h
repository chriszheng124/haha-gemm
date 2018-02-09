#ifndef GEMM_BLKSIZE_H
#define GEMM_BLKSIZE_H

#include "include/common.h"

#define NC 4096

#define KC 176
#define MC 216

// #define KC 176
// #define MC 108

// #define KC 352
// #define MC 432


#ifdef USE_4X6_R_BLOCK 
    #define NR 6 // KC must be multiple of 4, since vld need 128 aligned
#elif defined(USE_4X8_R_BLOCK)
    #define NR 8
#else 
    #define NR 4
#endif 

#define MR 4


#endif //GEMM_BLKSIZE_H
