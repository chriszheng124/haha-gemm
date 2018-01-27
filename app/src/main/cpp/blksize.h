#ifndef GEMM_BLKSIZE_H
#define GEMM_BLKSIZE_H

#define NC 4096

#define KC 176
#define MC 216

// #define KC 192
// #define MC 280

// #define KC 352
// #define MC 432

// #define NR 4
#define NR 6 // KC must be multiple of 4, since vld need 128 aligned
#define MR 4


#endif //GEMM_BLKSIZE_H
