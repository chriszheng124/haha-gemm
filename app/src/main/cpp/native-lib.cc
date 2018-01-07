#include <jni.h>
#include <string>
#include<iostream>

#include "utils.h"
#include "sgemm.h"
#include "log-streambuffer.h"
#include "include/blis/blis.h"

using namespace HahaGemm;

extern "C" int asm_abs(float* src, float* dest, int count);

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

    const int count = 1;
    float a[count] = {9.2};
    float b[count] = {};

    asm_abs(b, a, count);
    hello.append(std::to_string(b[0]));
    return env->NewStringUTF(hello.c_str());
}

extern "C" void
Java_com_haha_gemm_MainActivity_testGemm(JNIEnv* env, jobject object){
    int m = 512;
    int n = 512;
    int k = 512;
    int lda = m;
    int ldb = k;
    int ldc = m;

    float alpha = 2.0;
    float beta = 3.0;

    float* a;
    float* b;
    float* c = new float[m*n];
    memset(c, 0, sizeof(float)*m*n);

    a = new float[m*k];
    b = new float[k*n];
    Utils::MakeMatRandomly(a, m, k);
    //Utils::PrintMat(a, m, k, lda);

    Utils::MakeMatRandomly(b, k, n);
    //Utils::PrintMat(b, k, n, ldb);

    sgemm(false, false, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    // Utils::PrintMat(c, m, n, ldc);
    
#ifdef USE_BLIS
    // std::cout<<"blis:---------------------------------------------"<<std::endl;
    float* blis_c = new float[m*n];
    memset(blis_c, 0, sizeof(float)*m*n);

    long current_time = Utils::GetCurrentTimeMs();

    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha, a, 1, lda,
            b, 1, ldb, &beta, blis_c, 1, ldc, NULL);

    std::cout<<"blis Using time : "<<Utils::GetCurrentTimeMs() - current_time <<std::endl;
    // Utils::PrintMat(blis_c, m, n, ldc);

    delete[] blis_c;

#endif 

    delete[] c;
    delete[] a;
    delete[] b;
}

