
__kernel void sgemm(const __global int* A,
                    __read_only image2d_t Bi,
                    __global int* C,
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
        int4 a[8];
        int4 b[4];
        int4 c[8];

        for (int i = 0; i < 8; i++)
        {
            c[i] = 0;
        }

        int A_y_off = (gy << 3) * lda;

        for (int pos = 0; pos < k; pos += 4)
        {
            for (int i = 0; i < 4; i++)
            {
                b[i] = read_imagei(Bi, sampler, (int2)(gx, pos + i));
            }

            int A_off = A_y_off + pos;

            for (int i = 0; i < 8; i++)
            {
                a[i] = vload4(0, A + A_off);
                A_off += lda;
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
