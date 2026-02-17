#include <renderApi.hpp>
#include <iostream>

using namespace renderApi;

int main() {
    std::cout << "=== GPU Task Example - Static Pipelines ===" << std::endl;
    std::cout << std::endl;

    // ========== Step 1: Initialize Render API ==========
    std::cout << "[1] Initializing Render API..." << std::endl;
    
    QuickInitConfig initConfig;
    initConfig.appName = "GPUTask Example";
    initConfig.enableValidation = true;
    
    auto result = quickInit(initConfig);
    if (!result) {
        std::cerr << "Failed to initialize: " << result.errorMessage << std::endl;
        return 1;
    }
    
    auto instance = result.instance;
    auto gpu = result.gpu;
    std::cout << "✓ Instance and GPU created" << std::endl;
    std::cout << std::endl;

    // ========== Step 2: Create Static Pipelines ==========
    std::cout << "[2] Creating static pipelines..." << std::endl;
    
    // Pipeline 1: Compute shader for particle simulation
    PipelineConfig computeConfig;
    computeConfig.name = "ParticleSimulation";
    computeConfig.type = PipelineType::COMPUTE;
    computeConfig.compute.workGroupsX = 256;
    computeConfig.compute.workGroupsY = 1;
    computeConfig.compute.workGroupsZ = 1;
    
    // Add compute shader (STATIC - set at creation)
    ShaderConfig computeShader;
    computeShader.stage = ShaderStage::COMPUTE;
    computeShader.spirvCode = loadSPIRV("shaders/particle.comp.spv");
    computeConfig.shaders.push_back(computeShader);
    
    auto particlePipeline = createPipeline(gpu, computeConfig);
    if (particlePipeline == INVALID_PIPELINE) {
        std::cerr << "Failed to create particle pipeline" << std::endl;
        return 1;
    }
    std::cout << "✓ Created particle simulation pipeline" << std::endl;
    
    // Pipeline 2: Post-processing compute shader
    PipelineConfig postProcessConfig;
    postProcessConfig.name = "PostProcess";
    postProcessConfig.type = PipelineType::COMPUTE;
    postProcessConfig.compute.workGroupsX = 16;
    postProcessConfig.compute.workGroupsY = 16;
    
    ShaderConfig postProcessShader;
    postProcessShader.stage = ShaderStage::COMPUTE;
    postProcessShader.spirvCode = loadSPIRV("shaders/postprocess.comp.spv");
    postProcessConfig.shaders.push_back(postProcessShader);
    
    auto postProcessPipeline = createPipeline(gpu, postProcessConfig);
    std::cout << "✓ Created post-process pipeline" << std::endl;
    
    // Pipeline 3: Blur shader
    PipelineConfig blurConfig;
    blurConfig.name = "Blur";
    blurConfig.type = PipelineType::COMPUTE;
    
    ShaderConfig blurShader;
    blurShader.stage = ShaderStage::COMPUTE;
    blurShader.spirvCode = loadSPIRV("shaders/blur.comp.spv");
    blurConfig.shaders.push_back(blurShader);
    
    auto blurPipeline = createPipeline(gpu, blurConfig);
    std::cout << "✓ Created blur pipeline" << std::endl;
    std::cout << std::endl;

    // ========== Step 3: Create GPU Task ==========
    std::cout << "[3] Creating GPU Task..." << std::endl;
    
    GPUTaskConfig taskConfig;
    taskConfig.name = "MainRenderTask";
    taskConfig.autoRebuild = true;
    
    auto mainTask = createGPUTask(gpu, taskConfig);
    if (mainTask == INVALID_GPU_TASK) {
        std::cerr << "Failed to create GPU task" << std::endl;
        return 1;
    }
    std::cout << "✓ GPU Task created" << std::endl;
    std::cout << std::endl;

    // ========== Step 4: Register Pipelines to GPU Task ==========
    std::cout << "[4] Registering pipelines to GPU Task..." << std::endl;
    
    // Register pipelines in execution order
    int idx1 = registerPipeline(mainTask, particlePipeline, true, "Particle Sim");
    int idx2 = registerPipeline(mainTask, postProcessPipeline, true, "Post Process");
    int idx3 = registerPipeline(mainTask, blurPipeline, false, "Blur (disabled)");  // Start disabled
    
    std::cout << "✓ Registered pipeline [" << idx1 << "]: Particle Simulation (enabled)" << std::endl;
    std::cout << "✓ Registered pipeline [" << idx2 << "]: Post Process (enabled)" << std::endl;
    std::cout << "✓ Registered pipeline [" << idx3 << "]: Blur (disabled)" << std::endl;
    std::cout << std::endl;

    // ========== Step 5: Build and Execute ==========
    std::cout << "[5] Building and executing GPU Task..." << std::endl;
    
    if (!buildGPUTask(mainTask)) {
        std::cerr << "Failed to build GPU task" << std::endl;
        return 1;
    }
    std::cout << "✓ Command buffer built" << std::endl;
    
    if (!executeGPUTask(mainTask)) {
        std::cerr << "Failed to execute GPU task" << std::endl;
        return 1;
    }
    std::cout << "✓ GPU Task executed (particle sim + post process)" << std::endl;
    std::cout << std::endl;

    // ========== Step 6: Dynamic Control ==========
    std::cout << "[6] Dynamic pipeline control..." << std::endl;
    
    // Enable blur pipeline
    std::cout << "Enabling blur pipeline..." << std::endl;
    enablePipelineAt(mainTask, idx3);
    
    // Rebuild and execute
    rebuildGPUTask(mainTask);
    executeGPUTask(mainTask);
    std::cout << "✓ Executed with blur enabled" << std::endl;
    std::cout << std::endl;
    
    // Disable post-process temporarily
    std::cout << "Disabling post-process..." << std::endl;
    disablePipelineAt(mainTask, idx2);
    rebuildGPUTask(mainTask);
    executeGPUTask(mainTask);
    std::cout << "✓ Executed without post-process" << std::endl;
    std::cout << std::endl;

    // ========== Step 7: Reorder Pipelines ==========
    std::cout << "[7] Reordering pipelines..." << std::endl;
    
    // Move blur before post-process
    std::cout << "Moving blur to position 1..." << std::endl;
    movePipeline(mainTask, idx3, 1);
    
    // Re-enable all
    enableAllPipelines(mainTask);
    
    rebuildGPUTask(mainTask);
    executeGPUTask(mainTask);
    std::cout << "✓ Executed in new order: particle sim -> blur -> post process" << std::endl;
    std::cout << std::endl;

    // ========== Step 8: Query Information ==========
    std::cout << "[8] Task information..." << std::endl;
    
    auto taskInfo = getGPUTaskInfo(gpu, mainTask);
    std::cout << "Task name: " << taskInfo.name << std::endl;
    std::cout << "Total pipelines: " << taskInfo.pipelineCount << std::endl;
    std::cout << "Active pipelines: " << taskInfo.activePipelineCount << std::endl;
    std::cout << "Is built: " << (taskInfo.isBuilt ? "yes" : "no") << std::endl;
    std::cout << std::endl;
    
    auto pipelines = getRegisteredPipelines(mainTask);
    std::cout << "Registered pipelines (in order):" << std::endl;
    for (size_t i = 0; i < pipelines.size(); ++i) {
        std::cout << "  [" << i << "] " 
                  << (pipelines[i].enabled ? "✓" : "✗") << " "
                  << pipelines[i].name << std::endl;
    }
    std::cout << std::endl;

    // ========== Step 9: Unregister Pipeline ==========
    std::cout << "[9] Removing blur pipeline..." << std::endl;
    
    unregisterPipeline(mainTask, blurPipeline);
    std::cout << "✓ Blur pipeline unregistered" << std::endl;
    
    rebuildGPUTask(mainTask);
    executeGPUTask(mainTask);
    std::cout << "✓ Executed without blur" << std::endl;
    std::cout << std::endl;

    // ========== Step 10: Batch Operations ==========
    std::cout << "[10] Batch operations..." << std::endl;
    
    // Create multiple pipelines
    std::vector<std::vector<uint32_t>> shaders = {
        loadSPIRV("shaders/shader1.comp.spv"),
        loadSPIRV("shaders/shader2.comp.spv"),
        loadSPIRV("shaders/shader3.comp.spv")
    };
    
    auto batchPipelines = createMultipleComputePipelines(gpu, shaders, "BatchPipeline");
    std::cout << "✓ Created " << batchPipelines.size() << " pipelines" << std::endl;
    
    // Register them all at once
    size_t registered = registerMultiplePipelines(mainTask, batchPipelines, false);  // All disabled
    std::cout << "✓ Registered " << registered << " pipelines (all disabled)" << std::endl;
    std::cout << std::endl;

    // ========== Step 11: Cleanup ==========
    std::cout << "[11] Cleaning up..." << std::endl;
    
    // Destroy GPU task (automatically unregisters all pipelines)
    destroyGPUTask(gpu, mainTask);
    std::cout << "✓ GPU Task destroyed" << std::endl;
    
    // Pipelines remain valid and can be reused
    // Destroy pipelines
    destroyPipeline(gpu, particlePipeline);
    destroyPipeline(gpu, postProcessPipeline);
    destroyPipeline(gpu, blurPipeline);
    for (auto p : batchPipelines) {
        destroyPipeline(gpu, p);
    }
    std::cout << "✓ Pipelines destroyed" << std::endl;
    
    // Shutdown
    quickShutdown();
    std::cout << "✓ Render API shutdown" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Example Complete ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Key Concepts Demonstrated:" << std::endl;
    std::cout << "  • Static pipelines (created once with all shaders)" << std::endl;
    std::cout << "  • GPUTask as command buffer executor" << std::endl;
    std::cout << "  • Register/unregister pipelines dynamically" << std::endl;
    std::cout << "  • Enable/disable pipelines without destroying" << std::endl;
    std::cout << "  • Reorder pipeline execution" << std::endl;
    std::cout << "  • Batch operations" << std::endl;
    std::cout << "  • Automatic command buffer rebuild" << std::endl;
    
    return 0;
}