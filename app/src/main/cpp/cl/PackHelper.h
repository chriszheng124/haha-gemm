#ifndef HAHA_GEMM_PACKHELPER_H
#define HAHA_GEMM_PACKHELPER_H

#include <cstring>
#include "../base/Contsnts.h"

HAHA_GPU_BEGIN

template<typename T>
class PackHelper{
public:
    static void PackA(T* src,
                     int stride,
                     int src_col_num,
                     int src_row_num,
                     int k_step,
                     int local_size,
                     T* dest,
                     int dest_col_num,
                     int dest_row_num){
        int dest_pos = 0;

        int m_blk_num = src_row_num/local_size;
        int k_blk_num = src_col_num/k_step;

        for(int m = 0; m < m_blk_num; m++){
            for(int k = 0; k < k_blk_num; k++){
                for(int j = 0; j < local_size; ++j){
                    memcpy(dest + dest_pos,
                           src + k * k_step + stride * (m * local_size + j),
                           (size_t)k_step*sizeof(float));
                    dest_pos += k_step;
                }
            }

            int left_k_start = (src_col_num/k_step)*k_step;
            int left_k_len = src_col_num - left_k_start;
            if(left_k_len == 0){
                continue;
            }
            for(int j = 0; j < local_size; ++j){
                memcpy(dest + dest_pos,
                       src + left_k_start + stride * (m * local_size + j),
                       (size_t)left_k_len * sizeof(float));
                dest_pos += left_k_len;
                memset(dest + dest_pos, 0, (size_t)(dest_col_num - src_col_num) * sizeof(float));
                dest_pos += (dest_col_num - src_col_num);
            }
        }

        int left_m_start = (src_row_num/local_size)*local_size;
        int left_m_len = src_row_num - left_m_start;
        if(left_m_len == 0){
            return;
        }
        for(int k = 0; k < k_blk_num; k++){
            for(int j = left_m_start; j < left_m_len; ++j){
                memcpy(dest + dest_pos, src + k * k_step + stride * j,
                       (size_t)k_step * sizeof(float));
                dest_pos += k_step;
                memset(dest + dest_pos, 0,
                       (size_t)k_step * (dest_row_num - src_row_num) * sizeof(float));
                dest_pos += (k_step * (dest_row_num - src_row_num));
            }
        }

        int left_k_start = (src_col_num/k_step)*k_step;
        int left_k_len = src_col_num - left_k_start;
        if(left_k_len == 0){
            return;
        }
        for(int j = 0; j < left_m_len; ++j){
            memcpy(dest + dest_pos, src + left_k_start + stride * (j + left_m_start),
                   (size_t)left_k_len * sizeof(float));
            dest_pos += left_k_len;
            memset(dest + dest_pos, 0, (size_t)(dest_col_num - src_col_num) * sizeof(float));
            dest_pos += (dest_col_num - src_col_num);
        }
    };
};

HAHA_GPU_END

#endif //HAHA_GEMM_PACKHELPER_H
