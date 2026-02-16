#include <cstdint>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi::instance {

bool isInstanceExtensionAvailable(const char* extensionName) {
	uint32_t count = 0;
	if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS)
		throw std::runtime_error("Failed to get Vulkan instance extensions!");

	std::vector<VkExtensionProperties> extensions(count);
	if (vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()) != VK_SUCCESS)
		throw std::runtime_error("Failed to get Vulkan instance extensions!");

	for (const auto& ext : extensions)
		if (extensionName == std::string(ext.extensionName)) return true;
	return false;
}

}
