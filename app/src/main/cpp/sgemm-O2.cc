#include "sgemm.h"
#include "utils.h"
#include "blksize.h"


using namespace HahaGemm;

void sgemmO2(
        bool transa,
        bool transb,
        int m,
        int n,
        int k,
        float alpha,
        float* a,
        int lda,
        float* b,
        int ldb,
        float beta,
        float* c,
        int ldc){
    for(int jc = 0; jc < n; jc+=NC){
        for(int pc = 0; pc < k; pc+=KC){
            float* bc = new float[KC*NC];
            int bc_col_count = ((jc + NC) < n ? NC : n - jc);
            int bc_col_size = ((pc + KC) < k ? KC : k - pc); 
            float* b_start_pack = b + jc * ldb + pc; 
            int b_stride_pack = ldb - pc;
            for(int ic = 0; ic < m; ic+=MC){
                float* ac = new float[MC*KC];
                int ac_col_count = bc_col_size;
                int ac_col_size = ((ic + MC) < m ? MC : m - ic);
                float* a_start_pack = a + pc * lda + ic;
                int a_stride_pack = lda - ic;
            }
        }
    }
}
