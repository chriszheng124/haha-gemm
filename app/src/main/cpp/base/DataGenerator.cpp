#include <assert.h>
#include <cstdlib>
#include <ctime>
#include "DataGenerator.h"

void DataGenerator::GenerateFloatArray(float *data, uint length) {
    assert(data != NULL);

    srand((unsigned)time(0));
    int count = length;
    float base = 1.0f/(float)RAND_MAX;
    for(int i = 0; i < count; ++i){
        data[i] = (float)rand()*base;
    }
}

void DataGenerator::GenerateIntArray(int *data, uint length) {
    assert(data != NULL);

    srand((unsigned)time(0));
    for (int i = 0; i < length; ++i) {
        data[i] = rand();
    }
}
