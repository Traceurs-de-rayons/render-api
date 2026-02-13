#include "renderApi.hpp"

#include "renderInstance.hpp"

#include <exception>
#include <iostream>

using namespace renderApi;
using namespace renderApi::instance;

InitInstanceResult initNewInstance(const Config& config) {
	try {
		for (const char* extention : config.extensions)
			if (!isInstanceExtensionAvailable(extention)) return EXTENTIONS_NOT_AVAILABLE;
	} catch (const std::exception& e) {
		std::cerr << "Error initializing render instance: " << e.what() << std::endl;
		return VK_GET_EXTENTION_FAILED;
	}

	try {
		detail::instances_.emplace_back(config);
	} catch (const std::exception& e) {
		std::cerr << "Error initializing render instance: " << e.what() << std::endl;
		return VK_CREATE_INSTANCE_FAILED;
	}
	return INIT_VK_INSTANCESUCCESS;
}
