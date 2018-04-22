#ifndef HAHA_GEMM_DATAGENERATOR_H
#define HAHA_GEMM_DATAGENERATOR_H


#include <sys/types.h>

class DataGenerator {
public:
    static void GenerateFloatArray(float* data, uint length);

    static void GenerateIntArray(int* data, uint length);
};


#endif //HAHA_GEMM_DATAGENERATOR_H
