#include <jni.h>
#include <string>
#include<iostream>
#include <unistd.h>
#include <omp.h>

#include "utils.h"
#include "sgemm.h"
#include "log-streambuffer.h"
#include "test-data-factory.h"
#include "include/blis/blis.h"

using namespace HahaGemm;

LogStreamBuffer g_logBuffer;

jint JNI_OnLoad(JavaVM* vm, void* reserved){
    std::cout.rdbuf(&g_logBuffer);
#if defined(USE_BLIS)||defined(VERIFY_RESULT)
    bli_init();
#endif

    JNIEnv* env = NULL;
    jint result;

    if (vm->GetEnv((void**) &env, JNI_VERSION_1_4) != JNI_OK) {
        return -1;
    }
    result = JNI_VERSION_1_4;

    long page_size = sysconf(_SC_PAGESIZE);
    std::cout<<"page size = "<<page_size<<std::endl;
    std::cout<<"processor num = "<<omp_get_num_procs()<<std::endl;

    return result;
}

void JNI_UnLoad(JavaVM* vm, void* reserved){
#if defined(USE_BLIS)||defined(VERIFY_RESULT)
    bli_finalize();
#endif
}

extern "C" jstring
Java_com_haha_gemm_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    long page_size = sysconf(_SC_PAGESIZE);
    std::string hello = "Hello from _";
    hello.append(std::to_string(page_size));

    // verify cache miss rate...
    //sss
    return env->NewStringUTF(hello.c_str());
}

extern "C" void
Java_com_haha_gemm_MainActivity_testGemm(JNIEnv* env, jobject object){
#ifdef VERIFY_RESULT 
    int matric_count = 1;

    int m = 67;
    int n = 67;
    int k = 67;
    float* blis_result;

#else
    int matric_count = 500;
    int m = 512;
    int n = 512;
    int k = 1000;
#endif 
    int m_outer = m;//1024;
    int n_outer = n;//720;
    int k_outer = k;//720;
    
    int lda = m_outer;
    int ldb = k_outer;
    int ldc = m_outer;

    float alpha = 1.0;
    float beta = 1.0;

    TestDataFactory data_factory(10, m_outer, n_outer, k_outer);

    float* a;
    float* b;
    float* c;

#if defined(USE_BLIS)||defined(VERIFY_RESULT)
    for(int i = 0; i < matric_count; ++i){
        a = data_factory.GetA();
        b = data_factory.GetB();
        c = data_factory.GetC();

        memset(c, 0, sizeof(float)*m*n);

        long current_time = Utils::GetCurrentTimeMs();

        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha, a, 1, lda,
                b, 1, ldb, &beta, c, 1, ldc, NULL);

#ifdef VERIFY_RESULT
        blis_result = new float[m*n];
        memset(blis_result, 0, sizeof(float)*m*n);
        memcpy(blis_result, c, m*n*sizeof(float));
        Utils::PrintMat(c, m, n, ldc);
        //blis_result = Utils::PrintMatToString(c, m, n, ldc);
#else
        std::cout<<"blis Using time : "<<Utils::GetCurrentTimeMs() - current_time <<std::endl;
        data_factory.FreeA(a);
        data_factory.FreeB(b);
        data_factory.FreeC(c);
#endif 
    }
#endif 

#if defined(USE_LEVEL_O3)||defined(USE_LEVEL_O1)||defined(VERIFY_RESULT)||defined(USE_OMP) 
    for(int i = 0; i < matric_count; ++i){
#ifndef VERIFY_RESULT 
        a = data_factory.GetA();
        b = data_factory.GetB();
        c = data_factory.GetC();
#else 
        memset(c, 0, sizeof(float)*m*n);
#endif
        sgemm(false, false, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#ifdef VERIFY_RESULT
        Utils::PrintMat(c, m, n, ldc);
        //result = Utils::PrintMatToString(c, m, n, ldc);
        bool result = Utils::CompareMat(blis_result, c, m, n, ldc);
        if(result){
            std::cout<<"OK"<<std::endl;
        }else{
            std::cout<<"Error"<<std::endl;
        }
#endif
        data_factory.FreeA(a);
        data_factory.FreeB(b);
        data_factory.FreeC(c);
    }
#endif 
}

