#include "renderApi.hpp"

#include "renderDevice.hpp"
#include "renderInstance.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

using namespace renderApi;
using namespace renderApi::instance;

InitInstanceResult renderApi::initNewInstance(const Config& config) {
	try {
		for (const char* extention : config.extensions)
			if (!isInstanceExtensionAvailable(extention)) return EXTENTIONS_NOT_AVAILABLE;
	} catch (const std::exception& e) {
		std::cerr << "Error initializing render instance: " << e.what() << std::endl;
		return VK_GET_EXTENTION_FAILED;
	}

	try {
		getInstancesVector().emplace_back(config);
	} catch (const std::exception& e) {
		std::cerr << "Error initializing render instance: " << e.what() << std::endl;
		return VK_CREATE_INSTANCE_FAILED;
	}
	return INIT_VK_INSTANCE_SUCCESS;
}

std::vector<instance::RenderInstance>& renderApi::getInstances() {
	return getInstancesVector();
}

instance::RenderInstance* renderApi::getInstance(int index) {
	auto& instances = getInstancesVector();
	return instances.empty() ? nullptr : instances.size() ? &instances[index] : nullptr;
}

instance::RenderInstance* renderApi::getInstance(std::string name) {
	auto& instances = getInstancesVector();

	for (auto& instance : instances) {
		if (instance.getConfig().instanceName == name)
			return &instance;
	}

	return nullptr;
}

void renderApi::Api::cleanup() {
	getInstancesVector().clear();
}
