#include "buffer.hpp"
#include "gpuTask.hpp"
#include "renderApi.hpp"
#include "renderDevice.hpp"
#include "renderInstance.hpp"

#include <iostream>
#include <vulkan/vulkan_core.h>

int main() {
	INIT_RENDER_API;

	renderApi::instance::Config config;

	config.apiVersion	 = VK_API_VERSION_1_4;
	config.appVersion	 = VK_MAKE_VERSION(0, 0, 1);
	config.engineVersion = VK_MAKE_VERSION(0, 0, 1);
	config.appName		 = "RT - tkt frrr";
	config.engineName	 = "Engine jsp";
	config.extensions	 = {"VK_EXT_debug_utils"};
	config.layers		 = {"VK_LAYER_KHRONOS_validation"};
	config.instanceName	 = "compute";

	auto instanceResult = renderApi::initNewInstance(config);
	if (instanceResult != renderApi::instance::InitInstanceResult::INIT_VK_INSTANCE_SUCCESS) {
		std::cerr << "Error initializing render API: " << instanceResult << std::endl;
		return 1;
	}
	std::cerr << "render API initialized successfully" << std::endl;

	auto* instance = renderApi::getInstance(0);

	renderApi::device::Config deviceConfig;
	deviceConfig.graphics = 0;
	deviceConfig.compute = 1;
	deviceConfig.transfer = 0;
	deviceConfig.name = "compute";
	deviceConfig.physicalDevice = renderApi::device::selectBestPhysicalDevice(instance->getInstance());

	instance->addGPU(deviceConfig);
	renderApi::gpuTask::GpuTask task = renderApi::gpuTask::GpuTask("MainTask", instance->getGPU(0));

	renderApi::Buffer uni;

	uni.create(instance->getGPU(0), 10, renderApi::BufferType::UNIFORM);
	task.addBuffer(&uni, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	task.build();
	// instance.addGpuTask();

	// auto deviceResult = renderApi::addDevice(deviceConfig);
	// if (deviceResult != renderApi::device::InitDeviceResult::INIT_DEVICE_SUCCESS)
	// 	std::cerr << "Device initialization failed with code: " << deviceResult << std::endl;
	// std::cerr << "Device initialized successfully" << std::endl;

	// auto* gpu = renderApi::getGPU(0);
	// if (!gpu) {
	// 	std::cerr << "Failed to get GPU device" << std::endl;
	// 	return 1;
	// }

	// auto task = renderApi::gpuTask::GpuTask("MainTask", gpu);

	return 0;
}
