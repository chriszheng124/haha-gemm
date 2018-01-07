#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include "include/common.h"

HAHA_GEMM_BEGIN

class Utils{
public:
    static long GetCurrentTimeMs();
    static void MakeMatRandomly(float* mat, int m, int n);
    static void PrintMat(float* mat, int m, int n, int ld);

    static void PackA(int mc, int kc, const float* a,
            int inc_row, int inc_col, float* buffer);
    static void PackB(int kc, int nc, const float* b,
            int inc_row, int inc_col, float* buffer);

    static void Scale(int m, int n, float alpha, float* x,
            int inc_row, int inc_col);

    static void ScaleAdd(int m, int n, float alpha, const float* x,
            int inc_row_x, int inc_col_x, float* y, int inc_row_y, int inc_col_y);

private:
    static void PackMRxK(int k, const float* a, 
            int inc_row, int inc_col, float* buffer);
    static void PackKxNR(int k, const float* b,
            int inc_row, int inc_col, float* buffer);
};

HAHA_GEMM_END

#endif //GEMM_UTILS_H
