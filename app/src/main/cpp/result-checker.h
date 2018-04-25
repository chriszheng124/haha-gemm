#ifndef HAHA_GEMM_RESULTCHECKER_H
#define HAHA_GEMM_RESULTCHECKER_H


class ResultChecker {
public:
    template<typename T>
    static void sgemm_row_major(
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
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                int c_pos = i*ldc + j;
                T ab = 0;
                for(int p = 0; p < k; ++p){
                    ab += a[p + i * lda] * b[j + p * ldb];
                }
                c[c_pos] = (beta * c[c_pos] + alpha * ab);
            }
        }
    }
};


#endif //HAHA_GEMM_RESULTCHECKER_H
