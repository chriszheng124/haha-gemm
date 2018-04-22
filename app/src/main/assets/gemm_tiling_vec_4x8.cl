
__kernel void sgemm(const __global float* A,
                    const __global float* Bi,
                    __global float* C,
                    const int m,
                    const int n,
                    const int k){

    const int lda = n;
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

        int A_y_off = (gy << 3) * lda;
        int B_x_off = (gx << 2) * 4;

        for (int pos = 0; pos < k; pos += 4)
        {

            int B_off = B_x_off + pos * n;
            for (int i = 0; i < 4; i++)
            {
                b[i] = vload4(0, Bi + B_off);
                B_off += n;
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
