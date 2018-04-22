#include <cstring>
#include <float.h>
#include <math.h>
#include "VectorSum.h"
#include "../base/PerfUtil.h"
#include "../base/LogUtil.h"
#include "../cl-demo.h"

HAHA_GPU_BEGIN

#define USE_MEM_MAP

bool VectorSum::Sum2(int *a, int *b, uint length, int *c, const char *kernel_code,
                     float *run_time) {
    addArrays(a, b, c, length, kernel_code, run_time);
    return true;
}

#ifdef USE_MEM_MAP
bool VectorSum::Sum(float *a, float* b, uint length, float *c) {
    size_t global_work_size[1];
    size_t local_work_size[1];

    long start_time = PerfUtil::GetCurrentTimeMs();

    size_t kernel_work_group_size = GetMaxKernelWorkgroupSize();

    LogUtil::V("kernel_max_work_group_size = %d", kernel_work_group_size);

    local_work_size[0]= kernel_work_group_size;
    global_work_size[0] = kernel_work_group_size *
                          ((length>>2 + local_work_size[0] - 1) / kernel_work_group_size);

    size_t work_group_count = global_work_size[0] / local_work_size[0];

    LogUtil::V("work_group_size = %d, global_work_size = %d, work_group_count = %d",
               local_work_size[0], global_work_size[0], work_group_count);

    LogUtil::V("deciding work group size using time %ld",
               PerfUtil::GetCurrentTimeMs() - start_time);

    cl_int error = CL_FALSE;
    cl_mem arg1 = NULL;
    cl_mem arg2 = NULL;
    cl_mem arg3 = NULL;
    do{
        start_time = PerfUtil::GetCurrentTimeMs();
        // set args
        arg1 = clCreateBuffer(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                              4*sizeof(float)*global_work_size[0], NULL, NULL);
        if(arg1 == NULL){
            break;
        }
        arg2 = clCreateBuffer(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                              4*sizeof(float)*global_work_size[0], NULL, NULL);
        if(arg2 == NULL){
            break;
        }

        arg3 = clCreateBuffer(context_, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,
                              4*sizeof(float)*global_work_size[0], NULL, NULL);
        if(arg3 == NULL){
            break;
        }

        LogUtil::V("create buffer using time: %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();
        float* arg1_ptr = (float*) clEnqueueMapBuffer(command_queue_, arg1, CL_TRUE, CL_MAP_READ,
                                                      0, length*sizeof(float),
                                                      0, NULL, NULL, &error);
        if(CL_FAILED(error) || arg1_ptr == NULL){
            LogUtil::E("map buffer arg1 failed with error code %d", error);
            break;
        }
        memcpy(arg1_ptr, a, length*sizeof(float));

        float* arg2_ptr = (float*) clEnqueueMapBuffer(command_queue_, arg2, CL_TRUE, CL_MAP_READ,
                                                      0, length*sizeof(float),
                                                      0, NULL, NULL, &error);
        if(CL_FAILED(error) || arg2_ptr == NULL){
            LogUtil::E("map buffer arg2 failed with error code %d", error);
            break;
        }
        memcpy(arg2_ptr, b, length*sizeof(float));

        LogUtil::V("EnQueueMapBuffer using time: %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();

        error = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &arg1);
        error |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &arg2);
        error |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &arg3);
        if(CL_FAILED(error)){
            break;
        }

        LogUtil::V("set kernel arguments using time %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();
        error = clEnqueueNDRangeKernel(command_queue_, kernel_, 1, NULL,
                                       global_work_size, local_work_size, 0, NULL, NULL);
        if(CL_FAILED(error)){
            break;
        }
        error = clFinish(command_queue_);

        LogUtil::V("computing using time: %ld", PerfUtil::GetCurrentTimeMs() - start_time);

        if(CL_FAILED(error)){
            break;
        }

        start_time = PerfUtil::GetCurrentTimeMs();
        error = clEnqueueReadBuffer(command_queue_, arg3, CL_TRUE, 0,
                                    length*sizeof(float), c, 0, NULL, NULL);

        LogUtil::V("enqueue read buffer using time : %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

    }while (false);

    start_time = PerfUtil::GetCurrentTimeMs();

    if(arg1 != NULL){
        clReleaseMemObject(arg1);
    }
    if(arg2 != NULL){
        clReleaseMemObject(arg2);
    }
    if(arg3 != NULL){
        clReleaseMemObject(arg3);
    }

    LogUtil::V("release memory object using time %ld",
               PerfUtil::GetCurrentTimeMs() - start_time);

    return CL_SUCCEEDED(error);
}
#else
bool VectorSum::Sum(float *a, float* b, uint length, float *c) {
    size_t global_work_size[1];
    size_t local_work_size[1];

    long start_time = PerfUtil::GetCurrentTimeMs();

    size_t kernel_work_group_size = GetMaxKernelWorkgroupSize();

    LogUtil::V("kernel_max_work_group_size = %d", kernel_work_group_size);

    local_work_size[0]= kernel_work_group_size;
    global_work_size[0] = kernel_work_group_size *
            ((length>>2 + local_work_size[0] - 1) / kernel_work_group_size);

    size_t work_group_count = global_work_size[0] / local_work_size[0];

    LogUtil::V("work_group_size = %d, global_work_size = %d, work_group_count = %d",
               local_work_size[0], global_work_size[0], work_group_count);

    LogUtil::V("deciding work group size using time %ld",
               PerfUtil::GetCurrentTimeMs() - start_time);

    cl_int error = CL_FALSE;
    cl_mem arg1 = NULL;
    cl_mem arg2 = NULL;
    cl_mem arg3 = NULL;
    do{
        start_time = PerfUtil::GetCurrentTimeMs();
        // set args
        arg1 = clCreateBuffer(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                                     4*sizeof(float)*global_work_size[0], NULL, NULL);
        if(arg1 == NULL){
            break;
        }
        arg2 = clCreateBuffer(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                                     4*sizeof(float)*global_work_size[0], NULL, NULL);
        if(arg2 == NULL){
            break;
        }

        arg3 = clCreateBuffer(context_, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,
                              4*sizeof(float)*global_work_size[0], NULL, NULL);
        if(arg3 == NULL){
            break;
        }

        LogUtil::V("create buffer using time: %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();
        error = clEnqueueWriteBuffer(command_queue_, arg1, CL_FALSE, 0,
                                     length*sizeof(float), a, 0, NULL, NULL);
        if(CL_FAILED(error)){
            break;
        }

        error = clEnqueueWriteBuffer(command_queue_, arg2, CL_FALSE, 0,
                                     length*sizeof(float), b, 0, NULL, NULL);
        if(CL_FAILED(error)){
            break;
        }
        LogUtil::V("EnQueueWriteBuffer using time: %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();

        error = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &arg1);
        error |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &arg2);
        error |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &arg3);
        if(CL_FAILED(error)){
            break;
        }

        LogUtil::V("set kernel arguments using time %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();
        error = clEnqueueNDRangeKernel(command_queue_, kernel_, 1, NULL,
                                       global_work_size, local_work_size, 0, NULL, NULL);
        if(CL_FAILED(error)){
            break;
        }
        error = clFinish(command_queue_);

        LogUtil::V("computing using time: %ld", PerfUtil::GetCurrentTimeMs() - start_time);

        if(CL_FAILED(error)){
            break;
        }

        start_time = PerfUtil::GetCurrentTimeMs();
        error = clEnqueueReadBuffer(command_queue_, arg3, CL_TRUE, 0,
                            length*sizeof(float), c, 0, NULL, NULL);

        LogUtil::V("enqueue read buffer using time : %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

    }while (false);

    start_time = PerfUtil::GetCurrentTimeMs();

    if(arg1 != NULL){
        clReleaseMemObject(arg1);
    }
    if(arg2 != NULL){
        clReleaseMemObject(arg2);
    }
    if(arg3 != NULL){
        clReleaseMemObject(arg3);
    }

    LogUtil::V("release memory object using time %ld",
               PerfUtil::GetCurrentTimeMs() - start_time);

    return CL_SUCCEEDED(error);
}
#endif

bool VectorSum::ValidResult(int *a, int *b, uint length, int *c) {
    long start_time = PerfUtil::GetCurrentTimeMs();

    int* result = new int[length];

    for (int i = 0; i < length; ++i) {
        result[i] = a[i] + b[i];
    }

    LogUtil::V("vector sum [cpu] using time %ld", PerfUtil::GetCurrentTimeMs() - start_time);

    for (int i = 0; i < length; ++i) {
        int diff = fabs(result[i] - c[i]);
        if(diff > 0){
            delete[] result;
            return false;
        }
    }
    delete[] result;
    return true;
}

bool VectorSum::ValidResult(float *a, float *b, uint length, float *c) {
    long start_time = PerfUtil::GetCurrentTimeMs();

    float* result = new float[length];

    for (int i = 0; i < length; ++i) {
        result[i] = a[i] + b[i];
    }

    LogUtil::V("vector sum [cpu] using time %ld", PerfUtil::GetCurrentTimeMs() - start_time);

    for (int i = 0; i < length; ++i) {
        float diff = fabs(result[i] - c[i]);
        if(diff > FLT_EPSILON){
            delete[] result;
            return false;
        }
    }
    delete[] result;
    return true;
}

HAHA_GPU_END

