#ifndef HAHA_GEMM_RESULTCHECKER_H
#define HAHA_GEMM_RESULTCHECKER_H


class ResultChecker {
public:
    static void sgemm_row_major(
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
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                int c_pos = i*ldc + j;
                float ab = 0.0;
                for(int p = 0; p < k; ++p){
                    ab += a[p + i * lda] * b[j+ p * ldb];
                }
                c[c_pos] = (beta * c[c_pos] + alpha * ab);
            }
        }
    }
};


#endif //HAHA_GEMM_RESULTCHECKER_H
