
__kernel void sgemm(__read_only image2d_t A,
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

    if (((gx << 2) < n) && ((gy << 3) < m))
    {
        float4 a[8];
        float4 b[4];
        float4 c[8];

        for (int i = 0; i < 8; i++)
        {
            c[i] = 0.0f;
        }

        for (int pos = 0; pos < k; pos += 4)
        {
            for (int i = 0; i < 4; i++)
            {
                b[i] = read_imagef(Bi, sampler, (int2)(gx, pos + i));
            }

            for (int i = 0; i < 8; i++)
            {
                a[i] = read_imagef(A, sampler, (int2)(gx, pos + i));
            }

            for (int i = 0; i < 8; i++)
            {
                c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
            }
        }

        for (int i = 0; i < 8; i++)
        {
            int C_offs = ((gy << 3) + i) * ldc + (gx << 2);
            vstore4(c[i], 0, C + C_offs);
        }
    }
}
