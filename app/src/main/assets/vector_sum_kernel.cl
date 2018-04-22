__kernel void sum(__global float* a, __global float* b, __global float* c){
    uint global_id = get_global_id(0);

    float4 a4 = vload4(global_id, a);
    float4 b4 = vload4(global_id, b);

    float4 c4 = a4 + b4;

    vstore4(c4, global_id, c);
}
