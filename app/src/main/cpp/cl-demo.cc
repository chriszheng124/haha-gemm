#ifndef HAHA_CL_DEMO_H
#define HAHA_CL_DEMO_H 

#include <jni.h>
#include <CL/cl.h>
#include <android/asset_manager.h>
#include <android/log.h>
#include <time.h>
#include <malloc.h>
#include "base/LogUtil.h"
#include "base/PerfUtil.h"

#define MAX_PLATFORMS_COUNT     16
#define CL_SUCCEEDED(clErr) CL_SUCCESS==clErr
#define CL_FAILED(clErr) CL_SUCCESS!=clErr
#define LOG_TAG "oclDebug"


cl_device_id* GetDeviceId(cl_platform_id &platform){
    cl_uint numDevices = 0;
    cl_device_id *devices=NULL;
    cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices > 0) //GPU available.
    {
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }

    size_t info_size = 0;
    status = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &info_size);
    if(info_size > 0){
        char* value = (char*) malloc(info_size);
        status = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, info_size, value, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "deviceInfo: %s", value);
        size_t max_work_group_size = 0;
        status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(max_work_group_size), &max_work_group_size, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_work_group_size: %d", max_work_group_size);

        cl_uint max_cu_count = 0;
        status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(max_cu_count), &max_cu_count, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_cu_count: %d", max_cu_count);

        cl_uint max_work_item_count[3] = {0, 0, 0};
        status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                 sizeof(max_work_item_count), &max_work_item_count, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_work_item_count: %d", max_work_item_count[0]);

        cl_ulong max_global_mem_size = 0;
        status = clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE,
                                 sizeof(max_global_mem_size), &max_global_mem_size, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_global_mem_size: %llu MB", max_global_mem_size/(1024*1024));
        cl_ulong max_constant_mem_size = 0;
        status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                 sizeof(max_constant_mem_size), &max_constant_mem_size, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_constant_buffer_zise: %llu KB", max_constant_mem_size);
        size_t max_constant_arg_count = 0;
        status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CONSTANT_ARGS,
                                 sizeof(max_constant_arg_count), &max_constant_arg_count, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_constant_arg_count: %d", max_constant_arg_count);
        cl_ulong max_local_mem_size = 0;
        status = clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE,
                                 sizeof(max_local_mem_size), &max_local_mem_size, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_local_mem_size: %llu KB", max_local_mem_size);
        cl_uint max_local_mem_type = 0;
        status = clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_TYPE,
                                 sizeof(max_local_mem_type), &max_local_mem_type, NULL);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "max_local_mem_type: %d", max_local_mem_type);
    }

    return devices;
}

void addArrays(const int *arrayA, const int *arrayB, const int *Result, int length,
               const char* kernelCode, float* runTime)
{
    /*char *kernelCode = "__kernel void vadd(__global const int *a, __global const int *b, __global int *c)"
                                "{"
                                "    int gid = get_global_id(0);"
                                "    c[gid] = a[gid] + b[gid];"
                                "}";*/
    long start_time;
    cl_platform_id platform = 0;
    cl_device_type clDEviceType = CL_DEVICE_TYPE_GPU; // default
    cl_kernel kernel = 0;
    cl_command_queue cmd_queue = 0;
    cl_context context = 0;
    cl_mem memobjs[3];
    cl_program program = 0;
    cl_int clErr;
    unsigned long long startTime = 0, endTime = 0;
    size_t kernel_work_group_size = 0;
    size_t prefered_work_group_size = 0;

    // get current platform id, assuming there are no more than 16 platforms in the system
    cl_platform_id pPlatforms[MAX_PLATFORMS_COUNT] = { 0 };
    cl_uint uiPlatformsCount = 0;
    clErr = clGetPlatformIDs(MAX_PLATFORMS_COUNT, pPlatforms, &uiPlatformsCount);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "clErr: %d uiPlatformsCount: %d", clErr, uiPlatformsCount);
    if (CL_FAILED(clErr) || 0==uiPlatformsCount)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "ERROR: Failed to find any platform.");
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "clErr: %d uiPlatformsCount: %d", clErr, uiPlatformsCount);
        return;
    }

    // go through the available platform and select our (vendor = "Intel Corporation")
    size_t szPlatformInfoSize = 0;
    char * pcPlatformInfo = NULL;
    for (cl_uint ui=0; ui<uiPlatformsCount; ui++)
    {
        clErr = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_VENDOR, 0, NULL, &szPlatformInfoSize);
        if (CL_SUCCEEDED(clErr) && szPlatformInfoSize)
        {
            pcPlatformInfo = new char[szPlatformInfoSize];
            if (NULL != pcPlatformInfo)
            {
                clErr = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_VENDOR, szPlatformInfoSize, pcPlatformInfo, NULL);
                __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "pcPlatformInfo: %s", pcPlatformInfo);
                if (CL_SUCCEEDED(clErr))// && !strncmp(pcPlatformInfo, "Intel(R) Corporation", szPlatformInfoSize))
                {
                    platform = pPlatforms[ui];
                    delete[] pcPlatformInfo;
                    break;
                }
                delete[] pcPlatformInfo;
            }
        }
    }
    if (0 == platform)
    {
        // no platform found
        return;
    }

    cl_device_id* did = GetDeviceId(platform);

    //create context
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "create context");
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
                                          (cl_context_properties)NULL};

    context = clCreateContextFromType(properties, clDEviceType, NULL, NULL, &clErr);
    if (CL_FAILED(clErr) || 0 == context)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "clErr: %d - Failed to create context", clErr);
        return;
    }

    // get context's devices
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "get context's devices");
    cl_device_id device = 0;
    clErr = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (CL_FAILED(clErr) || 0==device)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "clErr: %d - Failed to get context info", clErr);
        clReleaseContext(context);
        return;
    }

    // create a command-queue
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "create a command-queue");
    cmd_queue  = clCreateCommandQueue(context, device, 0, NULL);
    if (cmd_queue == (cl_command_queue)0)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "clErr: %d - Failed to create command queue", clErr);
        goto release_context;
    }


    size_t global_work_size[1];
    size_t local_work_size[1];

    // allocate the buffer memory objects
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "allocate the buffer memory objects");

    memobjs[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * length, (void*)arrayA, NULL);
    if (memobjs[0] == (cl_mem)0)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Failed to create memobjs[0]");
        goto release_queue;
    }

    memobjs[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * length, (void*)arrayB, NULL);
    if (memobjs[1] == (cl_mem)0)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Failed to create memobjs[1]");
        goto release_mem0;
    }

    memobjs[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * length, NULL, NULL);
        if (memobjs[1] == (cl_mem)0)
        {
            goto release_mem1;
        }

    // create program
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "create program");
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelCode, NULL, &clErr);
    if (CL_FAILED(clErr) || 0 == program)
    {
        goto release_mem2;
    }

    // build program
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "build program");
    clErr = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (CL_FAILED(clErr))
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        goto release_program;
    }

    // create the kernel
    kernel = clCreateKernel(program, "sum", NULL);
    if (kernel == (cl_kernel)0)
    {
        goto release_program;
    }

    start_time = PerfUtil::GetCurrentTimeMs();
    // set the args values
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "set the args values");
    clErr = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &memobjs[0]);

    clErr |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &memobjs[1]);

    clErr |= clSetKernelArg(kernel, 2,sizeof(cl_mem), (void *) &memobjs[2]);

    if (CL_FAILED(clErr))
    {
        goto release_all;
    }

    clGetKernelWorkGroupInfo(kernel, did[0], CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &kernel_work_group_size, NULL);

    clGetKernelWorkGroupInfo(kernel, did[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(size_t), &prefered_work_group_size, NULL);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                        "preferred_work_group_size: %d", prefered_work_group_size);
    // set work-item dimensions
    global_work_size[0] = length/4;
    local_work_size[0]= kernel_work_group_size;
    // execute kernel
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "execute kernel");
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    startTime = (unsigned long long)(tp.tv_sec * 1000000000 + tp.tv_nsec);
    clErr = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_FAILED(clErr))
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "clErr: %d - Failed to execute kernel", clErr);
        goto release_all;
    }
    clErr = clFinish(cmd_queue);
    if (CL_FAILED(clErr))
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "clErr: %d - Failed to finish queue", clErr);
        goto release_all;
    }
    clock_gettime(CLOCK_MONOTONIC, &tp);
    endTime = (unsigned long long)(tp.tv_sec * 1000000000 + tp.tv_nsec);
    *runTime    = (endTime - startTime) / 1000000.0f;
    // read output Buffer
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "read output Buffer");

    clErr = clEnqueueReadBuffer(cmd_queue, memobjs[2], CL_TRUE, 0, length * sizeof(int), (void*)Result, 0, NULL, NULL);
    if (CL_FAILED(clErr))
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "clErr: %d - Failed to read output Buffer", clErr);
        goto release_all;
    }
    LogUtil::V("Sum2 gpu using time %ld", PerfUtil::GetCurrentTimeMs() - start_time);

    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "Done!");

    //release kernel, program, and memory objects
release_all:
    clReleaseKernel(kernel);
release_program:
    clReleaseProgram(program);
release_mem2:
    clReleaseMemObject(memobjs[2]);
release_mem1:
    clReleaseMemObject(memobjs[1]);
release_mem0:
    clReleaseMemObject(memobjs[0]);
release_queue:
    clReleaseCommandQueue(cmd_queue);
release_context:
    clReleaseContext(context);
    return;
}

extern "C" {
JNIEXPORT void JNICALL Java_com_haha_gemm_MainActivity_addArraysViaOpenCL(JNIEnv * env, jobject obj,
        jintArray arrayA, jintArray arrayB, jintArray Result, jstring kernelCode, jfloatArray runTime);
};

JNIEXPORT void JNICALL Java_com_haha_gemm_MainActivity_addArraysViaOpenCL(JNIEnv * env, jobject obj,
        jintArray arrayA, jintArray arrayB, jintArray Result, jstring kernelCode, jfloatArray runTime)
{
    int *c_arrayA = env->GetIntArrayElements(arrayA,NULL);
    int *c_arrayB = env->GetIntArrayElements(arrayB,NULL);
    int *c_Result = env->GetIntArrayElements(Result,NULL);
    float *c_runTime = env->GetFloatArrayElements(runTime, NULL);
    int length = env->GetArrayLength(arrayA);
    const char *nativeKernelCode = env->GetStringUTFChars(kernelCode,0);
    addArrays(c_arrayA,c_arrayB, c_Result, length, nativeKernelCode, c_runTime);
    env->ReleaseIntArrayElements(arrayA,c_arrayA,0);
    env->ReleaseIntArrayElements(arrayB,c_arrayB,0);
    env->ReleaseIntArrayElements(Result,c_Result,0);
    env->ReleaseFloatArrayElements(runTime,c_runTime,0);
}
#endif 

