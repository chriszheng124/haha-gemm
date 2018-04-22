//#include <cstring>
//#include "MatTranspose.h"
//
//MatTranspose::MatTranspose(const char *kernel_code) {
//    kernel_code_ = new char[strlen(kernel_code) + 1];
//    strcpy(kernel_code_, kernel_code);
//}
//
//bool MatTranspose::Transpose(float *input, float *output) {
//    if(!Compile(kernel_code_, kKernelFuncName)){
//        return false;
//    }
//
//    size_t global_work_size[2];
//    size_t local_work_size[2];
//
//    size_t kernel_work_group_size = GetMaxKernelWorkgroupSize();
//
//    local_work_size[0]= kernel_work_group_size;
//    global_work_size[0] = kernel_work_group_size * (length / kernel_work_group_size + 1);
//    size_t work_group_count = global_work_size[0] / local_work_size[0];
//
//    cl_int error = CL_FALSE;
//    cl_mem arg1 = NULL;
//    cl_mem arg2 = NULL;
//    do{
//        // set args
//        arg1 = clCreateBuffer(context_, CL_MEM_READ_ONLY,
//                              sizeof(float)*global_work_size[0], NULL, NULL);
//        if(arg1 == NULL){
//            break;
//        }
//        arg2 = clCreateBuffer(context_, CL_MEM_READ_WRITE,
//                              sizeof(float)*work_group_count, NULL, NULL);
//        if(arg2 == NULL){
//            break;
//        }
//
//        error = clEnqueueWriteBuffer(command_queue_, arg1, CL_TRUE, 0,
//                                     length*sizeof(float), input, 0, NULL, NULL);
//        if(CL_FAILED(error)){
//            break;
//        }
//
////        error = clEnqueueWriteBuffer(command_queue_, arg1, CL_FALSE, 0,
////                                     local_work_size[0]*sizeof(float),
////                                     partial_result, 0, NULL, NULL);
////        if(CL_FAILED(error)){
////            break;
////        }
//        error = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &arg1);
//        error |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &arg2);
//        if(CL_FAILED(error)){
//            break;
//        }
//
//        error = clEnqueueNDRangeKernel(command_queue_, kernel_, 1, NULL,
//                                       global_work_size, local_work_size, 0, NULL, NULL);
//        if(CL_FAILED(error)){
//            break;
//        }
//        error = clFinish(command_queue_);
//        if(CL_FAILED(error)){
//            break;
//        }
//        error = clEnqueueReadBuffer(command_queue_, arg2, CL_FALSE, 0,
//                                    local_work_size[0]*sizeof(float), partial_result, 0, NULL, NULL);
//
//
//    }while (false);
//
//    if(arg1 != NULL){
//        clReleaseMemObject(arg1);
//    }
//    if(arg2 != NULL){
//        clReleaseMemObject(arg2);
//    }
//    return CL_SUCCEEDED(error);
//}
//
