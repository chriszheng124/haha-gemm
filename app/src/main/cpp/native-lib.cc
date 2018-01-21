#include <jni.h>
#include <string>
#include<iostream>

#include "utils.h"
#include "sgemm.h"
#include "log-streambuffer.h"
#include "include/blis/blis.h"

using namespace HahaGemm;

LogStreamBuffer g_logBuffer;


jint JNI_OnLoad(JavaVM* vm, void* reserved){
    std::cout.rdbuf(&g_logBuffer);
#ifdef USE_BLIS
    bli_init();
#endif

    JNIEnv* env = NULL;
    jint result;

    if (vm->GetEnv((void**) &env, JNI_VERSION_1_4) != JNI_OK) {
        return -1;
    }
    result = JNI_VERSION_1_4;

    return result;
}

void JNI_UnLoad(JavaVM* vm, void* reserved){
#ifdef USE_BLIS
    bli_finalize();
#endif
}

extern "C" jstring
Java_com_haha_gemm_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from _";

    return env->NewStringUTF(hello.c_str());
}

void init_c(float* c, int len){
    for(int i = 0; i < len; ++i){
        c[i] = 0;
    }
}

extern "C" void
Java_com_haha_gemm_MainActivity_testGemm(JNIEnv* env, jobject object){
    int matric_count = 20;
    int m = 512;
    int n = 512;
    int k = 512;
    int lda = m;
    int ldb = k;
    int ldc = m;

    float alpha = 2.0;
    float beta = 3.0;

    float* a =  new float[m*k];
    float* b = new float[k*n];
    float* c = new float[m*n];

#ifdef USE_BLIS
    init_c(c, m*n);
    for(int i = 0; i < matric_count; ++i){
        Utils::MakeMatRandomly(a, m, k);
        //Utils::PrintMat(a, m, k, lda);

        Utils::MakeMatRandomly(b, k, n);
        //Utils::PrintMat(b, k, n, ldb);
        
        long current_time = Utils::GetCurrentTimeMs();

        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha, a, 1, lda,
                b, 1, ldb, &beta, c, 1, ldc, NULL);

        std::cout<<"blis Using time : "<<Utils::GetCurrentTimeMs() - current_time <<std::endl;
        // Utils::PrintMat(c, m, n, ldc);
    }
#else  
    init_c(c, m*n);
    for(int i = 0; i < matric_count; ++i){
        Utils::MakeMatRandomly(a, m, k);
        //Utils::PrintMat(a, m, k, lda);

        Utils::MakeMatRandomly(b, k, n);
        //Utils::PrintMat(b, k, n, ldb);
        
        sgemm(false, false, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        // Utils::PrintMat(c, m, n, ldc);
    }
#endif 
    delete[] c;
    delete[] a;
    delete[] b;
}

