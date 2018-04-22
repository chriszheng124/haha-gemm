#ifndef HAHA_GEMM_CLENGINE_H
#define HAHA_GEMM_CLENGINE_H


#include <CL/cl.h>
#include "../base/Contsnts.h"

HAHA_GPU_BEGIN

class CLEngine {
public:
    CLEngine();
    virtual ~CLEngine();

    bool Compile(const char* kernel_code,
                 const char* kernel_func_name,
                 const char* option = NULL);

protected:
    size_t GetMaxKernelWorkgroupSize();
    size_t GetPreferredWorkgroupSizeMultiple();

private:
    void GetDeviceInfo();
    bool CreateContext();
    bool CreateCommandQueue();
    bool BuildKernel(const char* kernel_code,
                     const char* kernel_func_name,
                     const char* option);

protected:
    cl_platform_id  platform_id_;
    cl_kernel kernel_;
    cl_command_queue command_queue_;
    cl_context context_;
    cl_program program_;
    cl_device_id device_id_;
};

#define CL_SUCCEEDED(_ERROR) (_ERROR == CL_SUCCESS)
#define CL_FAILED(_ERROR) (_ERROR != CL_SUCCESS)

HAHA_GPU_END

#endif //HAHA_GEMM_CLENGINE_H
