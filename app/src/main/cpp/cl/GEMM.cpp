#include <cstring>
#include <unistd.h>
#include "GEMM.h"
#include "../base/PerfUtil.h"
#include "../base/LogUtil.h"

HAHA_GPU_BEGIN

#define USE_IMAGE
#define USE_4X8_BLOCK
//#define USE_4X4_BLOCK

//#define USE_NAIVE

bool GEMM::Calc(bool transa,
                bool transb,
                int m,
                int n,
                int k,
                float alpha,
                float *a,
                int lda,
                float *b,
                int ldb,
                float beta,
                float *c,
                int ldc) {

    size_t global_work_size[2];
    size_t local_work_size[2];

    long start_time = PerfUtil::GetCurrentTimeMs();

    size_t kernel_work_group_size = GetMaxKernelWorkgroupSize();

    LogUtil::V("kernel_max_work_group_size = %d", kernel_work_group_size);

    int aligned_m = m;
    int aligned_n = n;
    int aligned_k = k;

#ifdef USE_4X8_BLOCK
    aligned_m = m % 8 == 0 ? m : m + 8;
    aligned_n = n % 4 == 0 ? n : n + 4;
    aligned_k = k % 4 == 0 ? k : k + 4;
#endif

#ifdef USE_NAIVE
    global_work_size[0] = (size_t)n;
#else
    global_work_size[0] = (size_t)aligned_n/4;
#endif

#ifdef USE_4X8_BLOCK
    global_work_size[1] = (size_t)aligned_m/8;
#elif defined(USE_4X4_BLOCK)
    global_work_size[1] = (size_t)(m + 3)/4;
#elif defined(USE_NAIVE)
    global_work_size[1] = (size_t)m;
#endif

#ifdef USE_NAIVE
    local_work_size[0] = 8;
    local_work_size[1] = 8;
#else
//    local_work_size[0] = 16;
//    local_work_size[1] = 64;
    local_work_size[0] = 4;
    local_work_size[1] = 8;
#endif

    LogUtil::V("work_group_size[0] = %d, work_group_size[1] = %d, "
               "global_work_size[0] = %d, global_work_size[1] = %d",
               local_work_size[0], local_work_size[1],
               global_work_size[0], global_work_size[1]);

    LogUtil::V("deciding work group size using time %ld",
               PerfUtil::GetCurrentTimeMs() - start_time);

    cl_int error = CL_FALSE;
    cl_mem arg1 = NULL;
    cl_mem arg2 = NULL;
    cl_mem arg3 = NULL;

    do{
        start_time = PerfUtil::GetCurrentTimeMs();

        arg1 = clCreateBuffer(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                              aligned_m*aligned_k*sizeof(float), NULL, NULL);
        if(arg1 == NULL){
            break;
        }
#ifdef USE_IMAGE
        cl_image_format image_format;
        image_format.image_channel_order = CL_RGBA;
        image_format.image_channel_data_type = CL_FLOAT;
        arg2 = clCreateImage2D(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                               &image_format, (size_t)aligned_n/4,
                               (size_t)aligned_k, 0, NULL, NULL);
#else
        arg2 = clCreateBuffer(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                              n*k*sizeof(float), NULL, NULL);
#endif
        if(arg2 == NULL){
            break;
        }
        arg3 = clCreateBuffer(context_, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
                              aligned_m*aligned_n*sizeof(float), NULL, NULL);
        if(arg3 == NULL){
            break;
        }

        LogUtil::V("create buffer using time: %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();
        float* arg1_ptr = (float*) clEnqueueMapBuffer(command_queue_, arg1, CL_TRUE, CL_MAP_READ,
                                                      0, aligned_m*aligned_k*sizeof(float),
                                                      0, NULL, NULL, &error);
        if(CL_FAILED(error) || arg1_ptr == NULL){
            LogUtil::E("map buffer arg1 failed with error code %d", error);
            break;
        }
        for(int i = 0; i < m; ++i){
            float* start = arg1_ptr + i * aligned_k;
            memcpy(start, a + i * k, k * sizeof(float));
            memset(start + k, 0, (aligned_k - k) * sizeof(float));
        }
        error = clEnqueueUnmapMemObject(command_queue_, arg1, arg1_ptr, 0, NULL, NULL);
        if(CL_FAILED(error)){
            LogUtil::E("unmap arg1 failed with error code %d", error);
            break;
        }
#ifdef USE_IMAGE
        size_t origin[3] = {0, 0, 0};
        size_t region[3];
        region[0] = (size_t)aligned_n/4;
        region[1] = (size_t)aligned_k;
        region[2] = 1;
        size_t row_pitch;
        size_t slice_pitch;
        float* arg2_ptr = (float*) clEnqueueMapImage(command_queue_, arg2, CL_TRUE, CL_MAP_READ,
                                                     origin, region, &row_pitch, &slice_pitch, 0,
                                                     NULL, NULL, NULL);

        int img_ldb = row_pitch/sizeof(float);
        for(int i = 0; i < k; ++i){
            float* start = arg2_ptr + i * img_ldb;
            memcpy(start, b + i * n, n * sizeof(float));
            memset(start + n, 0, (aligned_n - n) * sizeof(float));
        }

        LogUtil::V("row_pitch = %d pixels, slice_pitch = %d pixels",
                   row_pitch/(4*sizeof(float)), slice_pitch/(4*sizeof(float)));

#else
        float* arg2_ptr = (float*) clEnqueueMapBuffer(command_queue_, arg2, CL_TRUE, CL_MAP_READ,
                                                      0, k*n*sizeof(float),
                                                      0, NULL, NULL, &error);
        memcpy(arg2_ptr, b, k*n*sizeof(float));
#endif
        if(CL_FAILED(error) || arg2_ptr == NULL){
            LogUtil::E("map buffer arg2 failed with error code %d", error);
            break;
        }
        error = clEnqueueUnmapMemObject(command_queue_, arg2, arg2_ptr, 0, NULL, NULL);
        if(CL_FAILED(error)){
            LogUtil::E("unmap arg2 failed with error code %d", error);
            break;
        }

        LogUtil::V("EnQueueMapBuffer using time: %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();

        error = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &arg1);
        error |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &arg2);
        error |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &arg3);
        error |= clSetKernelArg(kernel_, 3, sizeof(int), &m);
        error |= clSetKernelArg(kernel_, 4, sizeof(int), &n);
        error |= clSetKernelArg(kernel_, 5, sizeof(int), &k);
        if(CL_FAILED(error)){
            break;
        }

        LogUtil::V("set kernel arguments using time %ld",
                   PerfUtil::GetCurrentTimeMs() - start_time);

        start_time = PerfUtil::GetCurrentTimeMs();
        cl_event event;
        error = clEnqueueNDRangeKernel(command_queue_, kernel_, 2, NULL,
                                       global_work_size, local_work_size, 0, NULL, &event);
        if(CL_FAILED(error)){
            LogUtil::E("compute failed with error code %d", error);
            break;
        }
        WaitForComplete(NULL);

        LogUtil::V("computing using time: %ld", PerfUtil::GetCurrentTimeMs() - start_time);

        if(CL_FAILED(error)){
            break;
        }

        start_time = PerfUtil::GetCurrentTimeMs();
        float* tmp_c = c;
        if(m != aligned_m || n != aligned_n){
            tmp_c = new float[aligned_m * aligned_n];
        }
        error = clEnqueueReadBuffer(command_queue_, arg3, CL_TRUE, 0,
                                    aligned_m*aligned_n*sizeof(float), tmp_c, 0, NULL, NULL);
        if(c != tmp_c){
            for(int i = 0; i < m; ++i){
                memcpy(c + i * n, tmp_c + i * aligned_n, n);
            }
            delete[] tmp_c;
        }

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

void GEMM::WaitForComplete(cl_event event) {
    if(event == NULL){
        clFinish(command_queue_);
        return;
    }
    size_t status;
    cl_event_info event_info = CL_EVENT_COMMAND_EXECUTION_STATUS;
    do{
        clGetEventInfo(event, event_info, sizeof(cl_event_info), &status, NULL);
        sleep(0);
    }while (status != CL_COMPLETE);
}

HAHA_GPU_END
