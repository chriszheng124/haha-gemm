#ifndef GEMM_TEST_DATA_FACTORY_H
#define GEMM_TEST_DATA_FACTORY_H

#include <list>
#include "include/common.h"
#include "utils.h"

HAHA_GEMM_BEGIN 

using namespace std;

template <typename T>
class TestDataFactory {
public:
    TestDataFactory(int data_set_num, int m, int n, int k){
        data_set_num_ = data_set_num;
        for(int i = 0; i < data_set_num_; ++i){
            T* a = new T[m*k];
            a_data_list_.push_back(a);
            Utils::MakeMatRandomly(a, m, k);

            T* b = new T[k*n];
            b_data_list_.push_back(b);
            Utils::MakeMatRandomly(b, k, n);

            T* c = new T[m*n];
            c_data_list_.push_back(c);
            memset(c, 0, m*n*sizeof(T));
        }
    }

    T* GetA(){
        if(a_data_list_.size() <= 0){
            return NULL;
        }
        T* data = a_data_list_.front();
        a_data_list_.pop_front();
        return data;
    }

    T* GetB(){
        if(b_data_list_.size() <= 0){
            return NULL;
        }
        T* data = b_data_list_.front();
        b_data_list_.pop_front();
        return data;
    }

    T* GetC(){
        if(c_data_list_.size() <= 0){
            return NULL;
        }
        T* data = c_data_list_.front();
        c_data_list_.pop_front();
        return data;
    }

    void FreeA(T* a){
        a_data_list_.push_back(a);
    }

    void FreeB(T* b){
        b_data_list_.push_back(b);
    }

    void FreeC(T* c){
        c_data_list_.push_back(c);
    }
    
    int GetDataSetCount(){
        return data_set_num_;
    }

private:
    int data_set_num_;

    list<T*> a_data_list_;
    list<T*> b_data_list_;
    list<T*> c_data_list_;
};

HAHA_GEMM_END 

#endif // GEMM_TEST_DATA_FACTORY_H

