
__kernel void sgemm(const __global float* A,
                    __read_only image2d_t Bi,
                    __global float* C,
                    const int m,
                    const int n,
                    const int k){

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|
                        CLK_FILTER_NEAREST|
                        CLK_ADDRESS_NONE;
    const int lda = n;
    const int ldc = n;

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (((gx << 2) < n) && ((gy << 3) < m))
    {
        float4 a[8];
        float4 b[4];
        float4 c[8];

        c[0] = 0.0f;
        c[1] = 0.0f;
        c[2] = 0.0f;
        c[3] = 0.0f;
        c[4] = 0.0f;
        c[5] = 0.0f;
        c[6] = 0.0f;
        c[7] = 0.0f;

        int A_y_off = (gy << 3) * lda;

        for (int pos = 0; pos < k; pos += 4)
        {
            b[0] = read_imagef(Bi, sampler, (int2)(gx, pos + 0));
            int A_off = A_y_off + pos;
            a[0] = vload4(0, A + A_off);
            A_off += lda;
            b[1] = read_imagef(Bi, sampler, (int2)(gx, pos + 1));
            a[1] = vload4(0, A + A_off);
            A_off += lda;
            b[2] = read_imagef(Bi, sampler, (int2)(gx, pos + 2));
            a[2] = vload4(0, A + A_off);
            A_off += lda;
            b[3] = read_imagef(Bi, sampler, (int2)(gx, pos + 3));

            a[3] = vload4(0, A + A_off);
            A_off += lda;
            a[4] = vload4(0, A + A_off);
            A_off += lda;
            a[5] = vload4(0, A + A_off);
            A_off += lda;
            a[6] = vload4(0, A + A_off);
            A_off += lda;
            a[7] = vload4(0, A + A_off);
            A_off += lda;

            c[0] += a[0].x * b[0] + a[0].y * b[1] + a[0].z * b[2] + a[0].w * b[3];
            c[1] += a[1].x * b[0] + a[1].y * b[1] + a[1].z * b[2] + a[1].w * b[3];
            c[2] += a[2].x * b[0] + a[2].y * b[1] + a[2].z * b[2] + a[2].w * b[3];
            c[3] += a[3].x * b[0] + a[3].y * b[1] + a[3].z * b[2] + a[3].w * b[3];
            c[4] += a[4].x * b[0] + a[4].y * b[1] + a[4].z * b[2] + a[4].w * b[3];
            c[5] += a[5].x * b[0] + a[5].y * b[1] + a[5].z * b[2] + a[5].w * b[3];
            c[6] += a[6].x * b[0] + a[6].y * b[1] + a[6].z * b[2] + a[6].w * b[3];
            c[7] += a[7].x * b[0] + a[7].y * b[1] + a[7].z * b[2] + a[7].w * b[3];
        }

        int gy_mul_8 = gy << 3;
        int gx_mul_4 = gx << 2;

        vstore4(c[0], 0, C + ((gy_mul_8) + 0) * ldc + (gx_mul_4));
        vstore4(c[1], 0, C + ((gy_mul_8) + 1) * ldc + (gx_mul_4));
        vstore4(c[2], 0, C + ((gy_mul_8) + 2) * ldc + (gx_mul_4));
        vstore4(c[3], 0, C + ((gy_mul_8) + 3) * ldc + (gx_mul_4));
        vstore4(c[4], 0, C + ((gy_mul_8) + 4) * ldc + (gx_mul_4));
        vstore4(c[5], 0, C + ((gy_mul_8) + 5) * ldc + (gx_mul_4));
        vstore4(c[6], 0, C + ((gy_mul_8) + 6) * ldc + (gx_mul_4));
        vstore4(c[7], 0, C + ((gy_mul_8) + 7) * ldc + (gx_mul_4));
    }
}
