#include "renderApi.hpp"
#include <iostream>

int main() {
	std::cout << "=== Unified Pipeline API ===" << std::endl;

	auto result = renderApi::quickInit();
	if (!result) {
		std::cerr << "Init failed: " << result.errorMessage << std::endl;
		return 1;
	}

	std::cout << "\n[COMPUTE Pipeline]" << std::endl;
	renderApi::PipelineConfig computeCfg;
	computeCfg.name = "ComputePipeline";
	computeCfg.type = renderApi::PipelineType::COMPUTE;
	computeCfg.compute.workGroupsX = 256;
	auto compute = renderApi::createPipeline(result.gpu, computeCfg);
	std::cout << "Created: " << compute << std::endl;

	std::cout << "\n[RAY_TRACING Pipeline]" << std::endl;
	renderApi::PipelineConfig rtCfg;
	rtCfg.name = "RayTracingPipeline";
	rtCfg.type = renderApi::PipelineType::RAY_TRACING;
	rtCfg.rayTracing.maxRecursionDepth = 10;
	auto rayTrace = renderApi::createPipeline(result.gpu, rtCfg);
	std::cout << "Created: " << rayTrace << std::endl;

	std::cout << "\n[MESH_SHADING Pipeline]" << std::endl;
	renderApi::PipelineConfig meshCfg;
	meshCfg.name = "MeshShadingPipeline";
	meshCfg.type = renderApi::PipelineType::MESH_SHADING;
	auto mesh = renderApi::createPipeline(result.gpu, meshCfg);
	std::cout << "Created: " << mesh << std::endl;

	std::cout << "\n[List All Pipelines]" << std::endl;
	auto all = renderApi::getPipelines(result.gpu);
	for (auto p : all) {
		auto info = renderApi::getPipelineInfo(result.gpu, p);
		const char* typeStr = "UNKNOWN";
		if (info.type == renderApi::PipelineType::COMPUTE) typeStr = "COMPUTE";
		else if (info.type == renderApi::PipelineType::RASTERIZATION) typeStr = "RASTERIZATION";
		else if (info.type == renderApi::PipelineType::RAY_TRACING) typeStr = "RAY_TRACING";
		else if (info.type == renderApi::PipelineType::MESH_SHADING) typeStr = "MESH_SHADING";
		std::cout << "  " << info.name << " [" << typeStr << "]" << std::endl;
	}

	std::cout << "\n[Cleanup]" << std::endl;

	std::cout << "Done" << std::endl;
	return 0;
}
