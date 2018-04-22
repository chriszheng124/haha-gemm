//
// Created by  zhengzhihui on 2018/4/17.
//

#ifndef HAHA_GEMM_CL_DEMO_H
#define HAHA_GEMM_CL_DEMO_H

void addArrays(const int *arrayA, const int *arrayB, const int *Result, int length,
                      const char* kernelCode, float* runTime);
#endif //HAHA_GEMM_CL_DEMO_H
