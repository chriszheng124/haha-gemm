
__kernel void sgemm(const __global float* A,
                    __read_only image2d_t Bi,
                    __global float* C,
                    const int m,
                    const int n,
                    const int k){

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|
                        CLK_FILTER_NEAREST|
                        CLK_ADDRESS_NONE;
    const int lda = k;
    const int ldc = n;

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (((gx << 3) < n) && ((gy << 3) < m))
    {
        float4 a[8];
        float4 b[8];
        float4 c[16];

        for (int i = 0; i < 16; i++)
        {
            c[i] = 0.0f;
        }

        int A_y_off = (gy << 3) * lda;

        for (int pos = 0; pos < k; pos += 4)
        {
            for (int i = 0; i < 4; i++)
            {
                b[2 * i] = read_imagef(Bi, sampler, (int2)(2*gx, pos + i));
                b[2 * i + 1] = read_imagef(Bi, sampler, (int2)(2*gx + 1, pos + i));
            }

            int A_off = A_y_off + pos;

            for (int i = 0; i < 8; i++)
            {
                a[i] = vload4(0, A + A_off);
                A_off += lda;
            }

            for (int i = 0; i < 4; i++)
            {
                c[2*i] += a[i].x * b[0] + a[i].y * b[2] + a[i].z * b[4] + a[i].w * b[6];
                c[2*i + 1] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[3] + a[i].w * b[5];
            }
        }

        for (int i = 0; i < 8; i++){
            int C_offs = ((gy << 3) + i) * ldc + (gx << 3);
            vstore4(c[2*i], 0, C + C_offs);
            vstore4(c[2*i + 1], 0, C + C_offs + 4);
        }
    }
}
