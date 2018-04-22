
__kernel void sgemm(const __global float* a,
                    const __global float* b,
                    __global float* c,
                    const int m,
                    const int n,
                    const int k){

    uint row = get_global_id(0);
    uint col = get_global_id(1);

    float r = 0.0f;
    uint col_x_k = col * k;
    for(int i = 0; i < k; ++i){
        r += (*(a + i * m + row)) * (*(b + i + col_x_k));
    }

    *(c + col * m + row) = r;
}