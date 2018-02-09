
void sgemmO1(
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
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < m; ++i){
            int c_pos = i + j * ldc;
            float ab = 0.0;
            int b_start = j * ldb;
            for(int p = 0; p < k; ++p){
                ab += a[p * lda + i] * b[b_start + p];
            }
            c[c_pos] = (beta * c[c_pos] + alpha * ab);
        }
    }
}
