#include "test-data-factory.h"
#include "utils.h"

HAHA_GEMM_BEGIN 

TestDataFactory::TestDataFactory(int data_set_num, int m, int n, int k){
    data_set_num_ = data_set_num;
    for(int i = 0; i < data_set_num_; ++i){
        float* a = new float[m*k];
        a_data_list_.push_back(a);
        Utils::MakeMatRandomly(a, m, k);

        float* b = new float[k*n];
        b_data_list_.push_back(b);
        Utils::MakeMatRandomly(b, k, n);

        float* c = new float[m*n];
        c_data_list_.push_back(c);
        memset(c, 0, m*n*sizeof(float));
        // Utils::MakeMatRandomly(c, m, n);
    }
}

float* TestDataFactory::GetA(){
    if(a_data_list_.size() <= 0){
        return NULL;
    }
    float* data = a_data_list_.front();
    a_data_list_.pop_front();
    return data;
}

float* TestDataFactory::GetB(){
    if(b_data_list_.size() <= 0){
        return NULL;
    }
    float* data = b_data_list_.front();
    b_data_list_.pop_front();
    return data;
}

float* TestDataFactory::GetC(){
    if(c_data_list_.size() <= 0){
        return NULL;
    }
    float* data = c_data_list_.front();
    c_data_list_.pop_front();
    return data;
}

void TestDataFactory::FreeA(float* a){
    a_data_list_.push_back(a);
}

void TestDataFactory::FreeB(float* b){
    b_data_list_.push_back(b);
}

void TestDataFactory::FreeC(float* c){
    c_data_list_.push_back(c);
}

HAHA_GEMM_END 

