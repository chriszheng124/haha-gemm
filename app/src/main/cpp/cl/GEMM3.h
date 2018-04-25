#ifndef HAHA_GEMM_GEMM3_H
#define HAHA_GEMM_GEMM3_H

#include <cstddef>
#include <cstring>
#include <unistd.h>
#include "../base/Contsnts.h"
#include "../base/PerfUtil.h"
#include "../base/LogUtil.h"

#include "CLEngine.h"

/*
 * a <-> image buffer
 * b <-> image buffer
 * c <-> mem buffer
 */

HAHA_GPU_BEGIN

template<typename T,
        cl_channel_type IMAGE_DATA_TYPE = CL_FLOAT,
        int BLK_SIZE_X = 4,
        int BLK_SIZE_Y = 8>
class GEMM3 : public CLEngine{
public:
    bool operator()(
            bool transa,
            bool transb,
            int m,
            int n,
            int k,
            T alpha,
            T* a,
            int lda,
            T* b,
            int ldb,
            T beta,
            T* c,
            int ldc){
        size_t global_work_size[2];
        size_t local_work_size[2];

        long start_time = PerfUtil::GetCurrentTimeMs();

        size_t kernel_work_group_size = GetMaxKernelWorkgroupSize();

        LogUtil::V("kernel_max_work_group_size = %d", kernel_work_group_size);

        int aligned_m = ((m + BLK_SIZE_Y - 1) / BLK_SIZE_Y) * BLK_SIZE_Y;
        int aligned_n = ((n + BLK_SIZE_X - 1) / BLK_SIZE_X) * BLK_SIZE_X;
        int aligned_k = ((k + 3) / 4) * 4;

        LogUtil::V("aligned_m = %d, aligned_n = %d, aligned_k = %d",
                   aligned_m, aligned_n, aligned_k);

        local_work_size[0] = 16;
        local_work_size[1] = 64;

        global_work_size[0] = ((aligned_n/BLK_SIZE_X + local_work_size[0] - 1)
                               / local_work_size[0]) * local_work_size[0];
        global_work_size[1] = ((aligned_m/BLK_SIZE_Y + local_work_size[1] - 1)
                               / local_work_size[1]) * local_work_size[1];

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

            cl_image_format a_image_format;
            a_image_format.image_channel_order = CL_RGBA;
            a_image_format.image_channel_data_type = IMAGE_DATA_TYPE;
            arg1 = clCreateImage2D(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                                   &a_image_format, (size_t)aligned_k/4,
                                   (size_t)aligned_m, 0, NULL, NULL);
            if(arg1 == NULL){
                break;
            }

            cl_image_format b_image_format;
            b_image_format.image_channel_order = CL_RGBA;
            b_image_format.image_channel_data_type = IMAGE_DATA_TYPE;
            arg2 = clCreateImage2D(context_, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                                   &b_image_format, (size_t)aligned_n/4,
                                   (size_t)aligned_k, 0, NULL, NULL);
            if(arg2 == NULL){
                break;
            }
            arg3 = clCreateBuffer(context_, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
                                  aligned_m*aligned_n*sizeof(T), NULL, NULL);
            if(arg3 == NULL){
                break;
            }

            LogUtil::V("create buffer using time: %ld",
                       PerfUtil::GetCurrentTimeMs() - start_time);

            start_time = PerfUtil::GetCurrentTimeMs();
            size_t a_origin[3] = {0, 0, 0};
            size_t a_region[3];
            a_region[0] = (size_t)aligned_k/4;
            a_region[1] = (size_t)aligned_m;
            a_region[2] = 1;
            size_t row_pitch;
            size_t slice_pitch;
            T* arg1_ptr = (T*) clEnqueueMapImage(command_queue_, arg1, CL_TRUE, CL_MAP_READ,
                                                 a_origin, a_region, &row_pitch, &slice_pitch, 0,
                                                 NULL, NULL, NULL);

            if(CL_FAILED(error) || arg1_ptr == NULL){
                LogUtil::E("map buffer arg1 failed with error code %d", error);
                break;
            }
            int img_lda = row_pitch/sizeof(T);
            for(int i = 0; i < m; ++i){
                T* start = arg1_ptr + i * img_lda;
                memcpy(start, a + i * k, k * sizeof(T));
                memset(start + k, 0, (aligned_k - k) * sizeof(T));
            }
            for(int i = m; i < aligned_m; ++i){
                memset(arg1_ptr + i * img_lda, 0, row_pitch);
            }

            error = clEnqueueUnmapMemObject(command_queue_, arg1, arg1_ptr, 0, NULL, NULL);
            if(CL_FAILED(error)){
                LogUtil::E("unmap arg1 failed with error code %d", error);
                break;
            }

            size_t b_origin[3] = {0, 0, 0};
            size_t b_region[3];
            b_region[0] = (size_t)aligned_n/4;
            b_region[1] = (size_t)aligned_k;
            b_region[2] = 1;
            size_t b_row_pitch;
            size_t b_slice_pitch;
            T* arg2_ptr = (T*) clEnqueueMapImage(command_queue_, arg2, CL_TRUE, CL_MAP_READ,
                                                         b_origin, b_region, &b_row_pitch,
                                                         &b_slice_pitch, 0,
                                                         NULL, NULL, NULL);

            int img_ldb = b_row_pitch/sizeof(T);
            for(int i = 0; i < k; ++i){
                T* start = arg2_ptr + i * img_ldb;
                memcpy(start, b + i * n, n * sizeof(T));
                memset(start + n, 0, (aligned_n - n) * sizeof(T));
            }
            for(int i = k; i < aligned_k; ++i){
                memset(arg2_ptr + i * img_ldb, 0, b_row_pitch);
            }

//            LogUtil::V("b_row_pitch = %d pixels, slice_pitch = %d pixels",
//                       b_row_pitch/(4*sizeof(T)), slice_pitch/(4*sizeof(T)));

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
            error |= clSetKernelArg(kernel_, 3, sizeof(int), &aligned_m);
            error |= clSetKernelArg(kernel_, 4, sizeof(int), &aligned_n);
            error |= clSetKernelArg(kernel_, 5, sizeof(int), &aligned_k);
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

            start_time = PerfUtil::GetCurrentTimeMs();
            T* tmp_c = c;
            if(m != aligned_m || n != aligned_n){
                tmp_c = new T[aligned_m * aligned_n];
            }
            error = clEnqueueReadBuffer(command_queue_, arg3, CL_TRUE, 0,
                                        aligned_m*aligned_n*sizeof(T), tmp_c, 0, NULL, NULL);
            if(c != tmp_c){
                for(int i = 0; i < m; ++i){
                    memcpy(c + i * n, tmp_c + i * aligned_n, n * sizeof(T));
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


    void WaitForComplete(cl_event event) {
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
};

HAHA_GPU_END

#endif //HAHA_GEMM_GEMM3_H
