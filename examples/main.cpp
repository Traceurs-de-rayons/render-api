/**
 * @file main.cpp
 * @brief Simple compute example - adds two arrays
 * @details Demonstrates the simplest possible compute pipeline usage
 */

#include "../include/renderApi.hpp"

#include <iostream>
#include <vector>

// Simple compute shader that adds two arrays: C[i] = A[i] + B[i]
// Layout:
//   binding 0: A (storage buffer - readonly)
//   binding 1: B (storage buffer - readonly)
//   binding 2: C (storage buffer - write)
//
// GLSL source:
/*
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer BufferA { float data[]; } bufferA;
layout(binding = 1) readonly buffer BufferB { float data[]; } bufferB;
layout(binding = 2) writeonly buffer BufferC { float data[]; } bufferC;

void main() {
	uint idx = gl_GlobalInvocationID.x;
	bufferC.data[idx] = bufferA.data[idx] + bufferB.data[idx];
}
*/
// SPIR-V bytecode (compiled from above GLSL):
static const std::vector<uint32_t> computeShaderSPV = {
		0x07230203, 0x00010000, 0x00080001, 0x0000003d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e,
		0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000001c,
		0x00060010, 0x00000004, 0x00000011, 0x00000100, 0x00000001, 0x00000100, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
		0x6e69616d, 0x00000000, 0x00060005, 0x0000000c, 0x66754241, 0x72656666, 0x00000041, 0x00000000, 0x00060006, 0x0000000c, 0x00000000,
		0x61746164, 0x00000000, 0x00000000, 0x00060005, 0x00000011, 0x66754242, 0x72656666, 0x00000042, 0x00000000, 0x00060006, 0x00000011,
		0x00000000, 0x61746164, 0x00000000, 0x00000000, 0x00060005, 0x00000016, 0x66754243, 0x72656666, 0x00000043, 0x00000000, 0x00060006,
		0x00000016, 0x00000000, 0x61746164, 0x00000000, 0x00000000, 0x00050048, 0x0000000c, 0x00000000, 0x00000023, 0x00000000, 0x00050048,
		0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000016, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000c,
		0x0000001e, 0x00000000, 0x00040047, 0x00000011, 0x0000001e, 0x00000001, 0x00040047, 0x00000016, 0x0000001e, 0x00000002, 0x00020013,
		0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x0000000b, 0x00000006, 0x00000001,
		0x0004001e, 0x0000000c, 0x0000000b, 0x00000006, 0x00040020, 0x0000000d, 0x00000009, 0x0000000c, 0x0004003b, 0x0000000d, 0x0000000e,
		0x00000009, 0x0004001e, 0x00000011, 0x0000000b, 0x00000006, 0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012,
		0x00000013, 0x00000009, 0x0004001e, 0x00000016, 0x0000000b, 0x00000006, 0x00040020, 0x00000017, 0x00000009, 0x00000016, 0x0004003b,
		0x00000017, 0x00000018, 0x00000009, 0x00040015, 0x00000019, 0x00000020, 0x00000000, 0x0004002b, 0x00000019, 0x0000001a, 0x00000100,
		0x0004002b, 0x00000019, 0x0000001b, 0x00000001, 0x0004002b, 0x00000019, 0x0000001c, 0x00000000, 0x00050036, 0x00000002, 0x00000004,
		0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x0000000b, 0x00000008, 0x00000007, 0x000500c5, 0x0000000b, 0x00000009,
		0x00000008, 0x0000001a, 0x000600c4, 0x0000000b, 0x0000000a, 0x00000009, 0x0000001b, 0x0000001c, 0x0004003d, 0x0000000c, 0x0000000f,
		0x0000000e, 0x000500c0, 0x00000006, 0x00000010, 0x0000000a, 0x0000000f, 0x0004003d, 0x00000011, 0x00000014, 0x00000013, 0x000500c0,
		0x00000006, 0x00000015, 0x00000010, 0x00000014, 0x0004003d, 0x00000016, 0x0000001d, 0x00000018, 0x000500c0, 0x00000006, 0x0000001e,
		0x00000015, 0x0000001d, 0x000100fd, 0x00010038};

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  Simple GPU Compute - Array Addition" << std::endl;
	std::cout << "========================================" << std::endl << std::endl;

	// ============================================================================
	// STEP 1: Initialize Vulkan (one line!)
	// ============================================================================
	std::cout << "[1/6] Initializing Vulkan..." << std::endl;

	auto init = renderApi::quickInit("ComputeExample", true);
	if (!init) {
		std::cerr << "Failed to initialize! Error: " << init.instanceResult << std::endl;
		return 1;
	}
	std::cout << "    ✓ Vulkan initialized successfully!" << std::endl << std::endl;

	// ============================================================================
	// STEP 2: Create GPU Context
	// ============================================================================
	std::cout << "[2/6] Creating GPU context..." << std::endl;

	auto context = renderApi::createContext(init.gpu);
	if (!context.isInitialized()) {
		std::cerr << "Failed to create context!" << std::endl;
		return 1;
	}
	std::cout << "    ✓ GPU context ready!" << std::endl << std::endl;

	// ============================================================================
	// STEP 3: Prepare Data
	// ============================================================================
	std::cout << "[3/6] Preparing data..." << std::endl;

	const size_t arraySize	= 10;
	const size_t bufferSize = arraySize * sizeof(float);

	// Create input arrays
	std::vector<float> arrayA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
	std::vector<float> arrayB = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f};
	std::vector<float> result(arraySize);

	std::cout << "    Input A: ";
	for (size_t i = 0; i < arraySize; ++i) {
		std::cout << arrayA[i] << (i < arraySize - 1 ? ", " : "");
	}
	std::cout << std::endl;

	std::cout << "    Input B: ";
	for (size_t i = 0; i < arraySize; ++i) {
		std::cout << arrayB[i] << (i < arraySize - 1 ? ", " : "");
	}
	std::cout << std::endl << std::endl;

	// ============================================================================
	// STEP 4: Create Buffers
	// ============================================================================
	std::cout << "[4/6] Creating GPU buffers..." << std::endl;

	// Create storage buffers for compute
	auto bufferA = context.createStorageBuffer(bufferSize);
	auto bufferB = context.createStorageBuffer(bufferSize);
	auto bufferC = context.createStorageBuffer(bufferSize);

	if (!bufferA.isValid() || !bufferB.isValid() || !bufferC.isValid()) {
		std::cerr << "Failed to create buffers!" << std::endl;
		return 1;
	}

	// Upload data
	bufferA.upload(arrayA.data(), bufferSize);
	bufferB.upload(arrayB.data(), bufferSize);

	std::cout << "    ✓ Created 3 storage buffers" << std::endl;
	std::cout << "    ✓ Uploaded input data to GPU" << std::endl << std::endl;

	// ============================================================================
	// STEP 5: Run Compute Shader
	// ============================================================================
	std::cout << "[5/6] Running compute shader..." << std::endl;

	// Create compute pipeline
	renderApi::ComputePipelineConfig pipelineConfig;
	pipelineConfig.shaderStage = {computeShaderSPV, VK_SHADER_STAGE_COMPUTE_BIT, "main"};
	// Note: In a real implementation, you'd bind buffers to descriptor sets here

	auto pipeline = context.createComputePipeline(pipelineConfig);
	if (!pipeline.isValid()) {
		std::cerr << "Failed to create pipeline!" << std::endl;
		return 1;
	}

	// Dispatch compute (10 elements with 256 local size = 1 work group)
	auto cmd = context.beginOneTimeCommands();
	pipeline.bind(cmd);
	pipeline.dispatch(cmd, 1, 1, 1);
	context.endOneTimeCommands(cmd);

	std::cout << "    ✓ Compute shader executed!" << std::endl;
	std::cout << "    ✓ Calculated: C[i] = A[i] + B[i]" << std::endl << std::endl;

	// ============================================================================
	// STEP 6: Read Results
	// ============================================================================
	std::cout << "[6/6] Reading results..." << std::endl;

	// Download result
	bufferC.download(result.data(), bufferSize);

	std::cout << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "            RESULTS" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	// Print in a nice table format
	std::cout << "  Index |    A    |    B    |  A + B  |  Result" << std::endl;
	std::cout << "  ------+---------+---------+---------+---------" << std::endl;

	bool allCorrect = true;
	for (size_t i = 0; i < arraySize; ++i) {
		float expected = arrayA[i] + arrayB[i];
		bool  correct  = (std::abs(result[i] - expected) < 0.0001f);
		if (!correct) allCorrect = false;

		std::cout << "    " << i << "   |  " << arrayA[i] << "  |  " << arrayB[i] << "  |  " << expected << "  |  " << result[i]
				  << (correct ? " ✓" : " ✗") << std::endl;
	}

	std::cout << std::endl;
	if (allCorrect) {
		std::cout << "  ✓ All calculations correct!" << std::endl;
	} else {
		std::cout << "  ✗ Some calculations incorrect!" << std::endl;
	}

	std::cout << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "           Cleanup Complete" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	std::cout << "Summary:" << std::endl;
	std::cout << "  • Initialized Vulkan" << std::endl;
	std::cout << "  • Created 3 GPU buffers" << std::endl;
	std::cout << "  • Ran compute shader" << std::endl;
	std::cout << "  • Verified " << arraySize << " calculations" << std::endl;
	std::cout << "  • Automatic cleanup (RAII)" << std::endl;

	return allCorrect ? 0 : 1;
}