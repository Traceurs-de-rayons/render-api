#include "renderApi.hpp"
#include <iostream>

int main() {
	std::cout << "=== Ray Tracing Pipeline Example ===" << std::endl;

	auto result = renderApi::quickInit();
	if (!result) return 1;

	std::cout << "\n[1] Create Ray Tracing Pipeline..." << std::endl;
	renderApi::PipelineConfig rtConfig;
	rtConfig.name = "RayTracingPipeline";
	rtConfig.type = renderApi::PipelineType::RAY_TRACING;
	rtConfig.rayTracing.maxRecursionDepth = 3;
	rtConfig.rayTracing.maxRayPayloadSize = 64;
	rtConfig.rayTracing.maxRayHitAttributeSize = 32;

	auto rtPipeline = renderApi::createPipeline(result.gpu, rtConfig);
	std::cout << "Pipeline created: " << rtPipeline << std::endl;

	std::cout << "\n[2] Add Shaders..." << std::endl;

	auto raygenSpirv = renderApi::loadSPIRV("shaders/raygen.spv");
	auto closesthitSpirv = renderApi::loadSPIRV("shaders/closesthit.spv");
	auto missSpirv = renderApi::loadSPIRV("shaders/miss.spv");
	auto anyhitSpirv = renderApi::loadSPIRV("shaders/anyhit.spv");

	if (raygenSpirv.empty()) {
		std::cout << "Shaders not found, using dummy..." << std::endl;
		raygenSpirv = {0x07230203};
		closesthitSpirv = {0x07230203};
		missSpirv = {0x07230203};
		anyhitSpirv = {0x07230203};
	}

	renderApi::ShaderConfig raygenCfg;
	raygenCfg.spirvCode = raygenSpirv;
	raygenCfg.stage = renderApi::ShaderStage::RAY_GENERATION;
	auto raygenShader = renderApi::addShader(rtPipeline, raygenCfg);
	std::cout << "Ray Generation shader: " << raygenShader << std::endl;

	renderApi::ShaderConfig closesthitCfg;
	closesthitCfg.spirvCode = closesthitSpirv;
	closesthitCfg.stage = renderApi::ShaderStage::CLOSEST_HIT;
	auto closesthitShader = renderApi::addShader(rtPipeline, closesthitCfg);
	std::cout << "Closest Hit shader: " << closesthitShader << std::endl;

	renderApi::ShaderConfig missCfg;
	missCfg.spirvCode = missSpirv;
	missCfg.stage = renderApi::ShaderStage::MISS;
	auto missShader = renderApi::addShader(rtPipeline, missCfg);
	std::cout << "Miss shader: " << missShader << std::endl;

	renderApi::ShaderConfig anyhitCfg;
	anyhitCfg.spirvCode = anyhitSpirv;
	anyhitCfg.stage = renderApi::ShaderStage::ANY_HIT;
	auto anyhitShader = renderApi::addShader(rtPipeline, anyhitCfg);
	std::cout << "Any Hit shader: " << anyhitShader << std::endl;

	std::cout << "\n[3] Create Shader Groups..." << std::endl;

	renderApi::RayTracingShaderGroup raygenGroup;
	raygenGroup.type = renderApi::RayTracingShaderGroupType::GENERAL;
	raygenGroup.generalShader = raygenShader;
	auto group0 = renderApi::addRayTracingShaderGroup(rtPipeline, raygenGroup);
	std::cout << "Group 0 (raygen): " << group0 << std::endl;

	renderApi::RayTracingShaderGroup missGroup;
	missGroup.type = renderApi::RayTracingShaderGroupType::GENERAL;
	missGroup.generalShader = missShader;
	auto group1 = renderApi::addRayTracingShaderGroup(rtPipeline, missGroup);
	std::cout << "Group 1 (miss): " << group1 << std::endl;

	renderApi::RayTracingShaderGroup hitGroup;
	hitGroup.type = renderApi::RayTracingShaderGroupType::TRIANGLES_HIT_GROUP;
	hitGroup.closestHitShader = closesthitShader;
	hitGroup.anyHitShader = anyhitShader;
	auto group2 = renderApi::addRayTracingShaderGroup(rtPipeline, hitGroup);
	std::cout << "Group 2 (hit group): " << group2 << std::endl;

	std::cout << "\n[4] Query Shader Groups..." << std::endl;
	auto groups = renderApi::getRayTracingShaderGroups(rtPipeline);
	std::cout << "Total shader groups: " << groups.size() << std::endl;

	std::cout << "\n[5] Pipeline Info..." << std::endl;
	auto info = renderApi::getPipelineInfo(result.gpu, rtPipeline);
	std::cout << "Name: " << info.name << std::endl;
	std::cout << "Shaders: " << info.shaderCount << std::endl;
	std::cout << "Built: " << info.isBuilt << std::endl;

	std::cout << "\n=== Example Complete ===" << std::endl;
	return 0;
}
