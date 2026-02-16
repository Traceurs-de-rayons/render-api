// compute.slang
// Compile with: slangc -target spirv -stage compute -entry main compute.slang -o compute.spv

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> bufferA;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float> bufferB;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> bufferC;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint idx = id.x;
    bufferC[idx] = bufferA[idx] + bufferB[idx];
}
