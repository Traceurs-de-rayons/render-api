#include <renderApi.hpp>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== Simple Compute Shader Example ===" << std::endl;
    std::cout << "Using simplified RenderApi helpers\n" << std::endl;

    // 1. Quick initialization (instance + device)
    auto initResult = renderApi::quickInit("ComputeExample", false);
    if (!initResult) {
        std::cerr << "Failed to initialize! Instance: " << initResult.instanceResult
                  << ", Device: " << initResult.deviceResult << std::endl;
        return -1;
    }
    std::cout << "✓ Vulkan initialized" << std::endl;

    // 2. Start GPU thread
    auto* gpu = renderApi::getGPU(0);
    gpu->running = true;
    std::cout << "✓ GPU thread started" << std::endl;

    // 3. Create GPU context
    auto context = renderApi::createContext(gpu);
    std::cout << "✓ GPU context created" << std::endl;

    // 4. Create output buffer (10,000 vec4 floats)
    const size_t elementCount = 10000;
    const size_t bufferSize = elementCount * sizeof(float) * 4;

    auto outputBuffer = context.createStorageBuffer(bufferSize, renderApi::BufferUsage::DYNAMIC);
    std::cout << "✓ Created output buffer (" << bufferSize << " bytes)" << std::endl;

    // 5. Load and create compute task
    auto shaderCode = renderApi::loadSPIRV("shaders/simple.comp.spv");
    if (shaderCode.empty()) {
        std::cerr << "Failed to load shader! Run: slangc shaders/simple.slang -target spirv -o shaders/simple.comp.spv -entry main -stage compute" << std::endl;
        return -1;
    }
    std::cout << "✓ Loaded shader (" << shaderCode.size() << " words)" << std::endl;

    auto* task = renderApi::createComputeTask(shaderCode, "GradientFill");
    task->bindBuffer(0, outputBuffer);
    task->setDispatchSize(40, 1, 1);  // 10000 / 256 = 40 groups

    if (!task->build()) {
        std::cerr << "Failed to build task!" << std::endl;
        return -1;
    }
    std::cout << "✓ Compute task built and running" << std::endl;

    // 6. Let it run for a few seconds
    std::cout << "\n=== Running for 3 seconds ===" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 7. Read back results
    std::cout << "\n=== Results ===" << std::endl;
    std::vector<float> results(elementCount * 4);
    outputBuffer.download(results.data(), bufferSize);

    std::cout << "First 5 elements (RGBA):" << std::endl;
    for (size_t i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] = ("
                  << results[i*4+0] << ", "
                  << results[i*4+1] << ", "
                  << results[i*4+2] << ", "
                  << results[i*4+3] << ")" << std::endl;
    }

    std::cout << "\nLast 5 elements (RGBA):" << std::endl;
    for (size_t i = elementCount - 5; i < elementCount; i++) {
        std::cout << "  [" << i << "] = ("
                  << results[i*4+0] << ", "
                  << results[i*4+1] << ", "
                  << results[i*4+2] << ", "
                  << results[i*4+3] << ")" << std::endl;
    }

    // 8. Cleanup
    std::cout << "\n=== Cleanup ===" << std::endl;
    gpu->running = false;
    if (gpu->finishCode.valid()) {
        gpu->finishCode.wait();
    }
    vkDeviceWaitIdle(gpu->device);
    outputBuffer.destroy();
    context.shutdown();
    std::cout << "✓ Clean exit" << std::endl;

    return 0;
}
