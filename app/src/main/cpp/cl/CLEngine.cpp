#include <malloc.h>
#include "CLEngine.h"
#include "../base/LogUtil.h"
#include "../base/PerfUtil.h"

HAHA_GPU_BEGIN

CLEngine::CLEngine() : kernel_(NULL),
                       context_(NULL),
                       command_queue_(NULL),
                       device_id_(NULL),
                       program_(NULL) {

}

CLEngine::~CLEngine() {
    if(kernel_ != NULL){
        clReleaseKernel(kernel_);
        kernel_ = NULL;
    }
    if(context_ != NULL){
        clReleaseContext(context_);
        context_ = NULL;
    }
    if(command_queue_ != NULL){
        clReleaseCommandQueue(command_queue_);
        command_queue_ = NULL;
    }
    if(program_ != NULL){
        clReleaseProgram(program_);
        program_ = NULL;
    }
    if(device_id_ != NULL){
        clReleaseDevice(device_id_);
        device_id_ = NULL;
    }
}

bool CLEngine::Compile(const char* kernel_code, const char* kernel_func_name, const char* option) {
    long start_time = PerfUtil::GetCurrentTimeMs();
    if(!CreateContext()){
        return false;
    }
    if(!CreateCommandQueue()){
        return false;
    }

    GetDeviceInfo();

    bool ret = BuildKernel(kernel_code, kernel_func_name, option);

    LogUtil::V("compile using time: %ld", PerfUtil::GetCurrentTimeMs() - start_time);

    return ret;
}

void CLEngine::GetDeviceInfo() {
    size_t info_size = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_NAME, 0, NULL, &info_size);
    if(info_size > 0){
        char* value = (char*) malloc(info_size);
        clGetDeviceInfo(device_id_, CL_DEVICE_NAME, info_size, value, NULL);
        LogUtil::V("connect to device : %s", value);
        free(value);
    }
    size_t max_work_group_size = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                             sizeof(max_work_group_size), &max_work_group_size, NULL);
    LogUtil::V("device_max_work_group_size = %d", max_work_group_size);

    cl_uint max_cu_count = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_COMPUTE_UNITS,
                             sizeof(max_cu_count), &max_cu_count, NULL);
    LogUtil::V("device_max_compute_unit_count = %d", max_cu_count);

    size_t max_work_item_count[3] = {0, 0, 0};
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                             sizeof(max_work_item_count), &max_work_item_count, NULL);
    LogUtil::V("device_max_work_item_count: [0] = %d, [1] = %d, [2] = %d",
               max_work_item_count[0], max_work_item_count[1], max_work_item_count[2]);

    cl_ulong max_global_mem_size = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_GLOBAL_MEM_SIZE,
                             sizeof(max_global_mem_size), &max_global_mem_size, NULL);
    LogUtil::V("device_global_memory_size = %lluMB", max_global_mem_size/(1024*1024));

    cl_ulong max_constant_mem_size = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                             sizeof(max_constant_mem_size), &max_constant_mem_size, NULL);
    cl_uint max_constant_arg_count = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_MAX_CONSTANT_ARGS,
                             sizeof(max_constant_arg_count), &max_constant_arg_count, NULL);
    cl_ulong max_local_mem_size = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_LOCAL_MEM_SIZE,
                             sizeof(max_local_mem_size), &max_local_mem_size, NULL);
    LogUtil::V("device_local_memory_size = %lluKB", max_local_mem_size/1024);

    cl_device_local_mem_type local_mem_type = 0;
    clGetDeviceInfo(device_id_, CL_DEVICE_LOCAL_MEM_TYPE,
                             sizeof(local_mem_type), &local_mem_type, NULL);
    LogUtil::V("local memory type is : %d", local_mem_type);

    size_t max_img_width;
    clGetDeviceInfo(device_id_, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                    sizeof(size_t), &max_img_width, NULL);
    LogUtil::V("device_max_image_width = %d", max_img_width);

    size_t max_img_height;
    clGetDeviceInfo(device_id_, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                    sizeof(size_t), &max_img_height, NULL);
    LogUtil::V("device_max_image_height = %d", max_img_height);

    cl_int preferred_vector_width;
    clGetDeviceInfo(device_id_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                    sizeof(cl_int), &preferred_vector_width, NULL);
    LogUtil::V("device_preferred_vector_width = %d", preferred_vector_width);
}

bool CLEngine::BuildKernel(const char* kernel_code,
                           const char* kernel_func_name,
                           const char* option) {
    cl_int error;
    program_ = clCreateProgramWithSource(context_, 1, &kernel_code, NULL, &error);
    if (CL_FAILED(error) || program_ == NULL){
        LogUtil::E("create program with source failed with errorcode %d", error);
        return false;
    }
    error = clBuildProgram(program_, 1, &device_id_, option, NULL, NULL);
    if (CL_FAILED(error)){
        size_t len;
        char buffer[1024] = {0};
        clGetProgramBuildInfo(program_, device_id_, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);

        LogUtil::E("build program failed with error info: %s", buffer);
        return false;
    }

    kernel_ = clCreateKernel(program_, kernel_func_name, &error);
    if(kernel_ == NULL || CL_FAILED(error)){
        LogUtil::E("create kernel failed with errorcode %d", error);
        return false;
    }
    return true;
}

bool CLEngine::CreateCommandQueue() {
    cl_int error;
    command_queue_ = clCreateCommandQueue(context_, device_id_, 0, &error);
    if(command_queue_ == NULL || CL_FAILED(error)){
        LogUtil::E("create command queue failed with error code %d", error);
        return false;
    }
    return true;
}

bool CLEngine::CreateContext() {
    cl_int error;
    cl_platform_id platforms[64] = {0};
    cl_uint platforms_count = 0;
    error = clGetPlatformIDs(64, platforms, &platforms_count);
    if (CL_FAILED(error) || 0 == platforms_count){
        LogUtil::E("get platform id failed with error code %d", error);
        return false;
    }

    size_t platform_info_size = 0;
    char* platform_info = NULL;
    for (int i = 0; i < platforms_count; ++i){
        error = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &platform_info_size);
        if (CL_SUCCEEDED(error) && platform_info_size> 0){
            platform_info = new char[platform_info_size];
            if (platform_info != NULL){
                error = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                                          platform_info_size, platform_info, NULL);
                if (CL_SUCCEEDED(error)){
                    LogUtil::V("query platform info %s", platform_info);
                    platform_id_ = platforms[i];
                }
                delete[] platform_info;
            }
        }
    }
    if (NULL == platform_id_){
        LogUtil::E("platform is NULL");
        return false;
    }

    size_t device_count;
    error = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 0, NULL, &device_count);
    if(CL_FAILED(error)){
        LogUtil::E("get device id failed with error code %d", error);
        return false;
    }
    LogUtil::V("get %d GPU device", device_count);
    clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, sizeof(cl_device_id), &device_id_, NULL);

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                          (cl_context_properties)platform_id_,
                                          (cl_context_properties) NULL};
    context_ = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
    if(context_ == NULL || CL_FAILED(error)){
        LogUtil::E("create context from type failed with error code %d", error);
        return false;
    }
    return true;
}

size_t CLEngine::GetMaxKernelWorkgroupSize() {
    size_t kernel_work_group_size;
    cl_int error = clGetKernelWorkGroupInfo(kernel_, device_id_, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &kernel_work_group_size, NULL);
    if(CL_FAILED(error)){
        LogUtil::E("get kernel workgroup info failed with error code %d", error);
    }
    return kernel_work_group_size;
}

size_t CLEngine::GetPreferredWorkgroupSizeMultiple() {
    size_t preferred_work_group_size;
    clGetKernelWorkGroupInfo(kernel_, device_id_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(size_t), &preferred_work_group_size, NULL);
    return preferred_work_group_size;
}

HAHA_GPU_END