__kernel void sum(__global int* a, __global int* b, __global int* c){
    uint global_id = get_global_id(0);

    uint id = 4 * global_id;
    int4 a4 = vload4(0, a + id);
    int4 b4 = vload4(0, b + id);

    int4 c4 = a4 + b4;

    vstore4(c4, 0, c + id);
}
