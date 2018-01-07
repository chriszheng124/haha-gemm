#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>

#include "utils.h"
#include "blksize.h"


HAHA_GEMM_BEGIN

long Utils::GetCurrentTimeMs(){
    timeval currTime;
    gettimeofday(&currTime, NULL);
    return currTime.tv_sec * 1000 + currTime.tv_usec/1000;
}

void Utils::MakeMatRandomly(float* mat, int m, int n){
    srand((unsigned)time(0));
    int count = m * n;
    for(int i = 0; i < count; ++i){
        mat[i] = rand()/(float)RAND_MAX;
    }
}

void Utils::PrintMat(float* mat, int m, int n, int ld){
    char buf[64];
    std::cout<<"Mat begin: row = "<<m<<"  col = "<<n<<std::endl;
    for(int i = 0; i < m; ++i){
        std::cout<<"\r\nrow "<<i<<": ";
        for(int j = 0; j < n; ++j){
            memset(buf, 0, sizeof(buf));
            sprintf(buf, "%.3f", mat[i + j * ld]);
            std::cout<<buf<<"   ";
        }
    }
    std::cout<<"\r\n----------------------Mat End-------------------------------"<<std::endl;
}

void Utils::PackA(int mc, int kc, const float* a,
        int inc_row, int inc_col, float* buffer){
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;

    for (i = 0; i < mp; ++i) {
        PackMRxK(kc, a, inc_row, inc_col, buffer);
        buffer += kc * MR;
        a      += MR * inc_row;
    }
    if (_mr > 0) {
        for (j = 0; j < kc; ++j) {
            for (i = 0; i < _mr; ++i) {
                buffer[i] = a[i * inc_row];
            }
            for (i = _mr; i < MR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR;
            a      += inc_col;
        }
    }
}

void Utils::PackB(int kc, int nc, const float* b,
        int inc_row, int inc_col, float* buffer){
    int np  = nc / NR;
    int _nr = nc % NR;

    int i, j;

    for (j = 0; j < np; ++j) {
        PackKxNR(kc, b, inc_row, inc_col, buffer);
        buffer += kc * NR;
        b      += NR * inc_col;
    }
    if (_nr>0) {
        for (i = 0; i < kc; ++i) {
            for (j = 0; j < _nr; ++j) {
                buffer[j] = b[j * inc_col];
            }
            for (j = _nr; j < NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            b      += inc_row;
        }
    }
}

void Utils::PackMRxK(int k, const float* a, 
        int inc_row, int inc_col, float* buffer){
    int i, j;

    for (j = 0; j < k; ++j) {
        for (i = 0; i < MR; ++i) {
            buffer[i] = a[i * inc_row];
        }
        buffer += MR;
        a      += inc_col;
    }
}

void Utils::PackKxNR(int k, const float* b,
        int inc_row, int inc_col, float* buffer){
    int i, j;

    for (i = 0; i < k; ++i) {
        for (j = 0; j < NR; ++j) {
            buffer[j] = b[j * inc_col];
        }
        buffer += NR;
        b      += inc_row;
    }
}

void Utils::Scale(int m, int n, float alpha, float* x,
        int inc_row, int inc_col){
    int i, j;

    if (alpha != 0.0) {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                x[i * inc_row + j * inc_col] *= alpha;
            }
        }
    } else {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                x[i*inc_row + j * inc_col] = 0.0;
            }
        }
    }
}

void Utils::ScaleAdd(int m, int n, float alpha, const float* x,
        int inc_row_x, int inc_col_x, float* y, int inc_row_y, int inc_col_y){
    int i, j;

    if (alpha != 1.0) {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                y[i * inc_row_y + j * inc_col_y] += alpha * x[i * inc_row_x + j * inc_col_x];
            }
        }
    } else {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
               y[i* inc_row_y + j * inc_col_y] +=x[i * inc_row_x + j * inc_col_x];
            }
        }
    }
}

HAHA_GEMM_END

