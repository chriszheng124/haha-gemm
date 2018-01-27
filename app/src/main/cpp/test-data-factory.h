#ifndef GEMM_TEST_DATA_FACTORY_H
#define GEMM_TEST_DATA_FACTORY_H

#include <list>
#include "include/common.h"

HAHA_GEMM_BEGIN 

using namespace std;

class TestDataFactory {
public:
    TestDataFactory(int data_set_num, int m, int n, int k);

    float* GetA();
    float* GetB();
    float* GetC();

    void FreeA(float* a);
    void FreeB(float* b);
    void FreeC(float* c);
    
    int GetDataSetCount(){
        return data_set_num_;
    }

private:
    int data_set_num_;

    list<float*> a_data_list_;
    list<float*> b_data_list_;
    list<float*> c_data_list_;
};

HAHA_GEMM_END 

#endif // GEMM_TEST_DATA_FACTORY_H

