
__kernel void sgemm(const __global float* a,
                    const __global float* b,
                    __global float* c,
                    const int m,
                    const int n,
                    const int k){
    const uint global_row = get_global_id(0);
    const uint global_col = get_global_id(1);

    const uint local_row = get_local_id(0);
    const uint local_col = get_local_id(1);

    __local float a_block[TS*TS];
    __local float b_block[TS*TS];

    float acc = 0.0f;
    const uint tile_num = k/TS;
    for(int i = 0; i < tile_num; ++i){
        const uint tile_row = i * TS + local_row;
        const uint tile_col = i * TS + local_col;

        *(a_block + local_col * TS + local_row) = *(a + tile_col * m + global_row);
        *(b_block + local_col * TS + local_row) = *(b + tile_row + global_col * k);

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int t = 0; t < TS; ++t){
            acc += (*(a_block + t * TS + local_row)) * (*(b_block + t + local_col * TS));
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    *(c + global_col * m + global_row) = acc;
}
