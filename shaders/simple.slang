// Simple compute shader that sets buffer values
// Each thread writes its global ID to the output buffer

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID, uniform RWStructuredBuffer<float4> outputBuffer) {
	uint index = dispatchThreadID.x;

	// Write a gradient color based on thread ID
	float t				= float(index) / 10000.0;
	outputBuffer[index] = float4(t, 1.0 - t, 0.5, 1.0);
}
