#ifndef UTILS_HPP
# define UTILS_HPP

# include <cstddef>
# include <string>
# include <vulkan/vulkan_core.h>

namespace renderApi {
	class Buffer;
}

std::string	   generateRandomString(size_t length = 8);
VkShaderModule createShaderModule(VkDevice device, const std::string& path);

bool saveBufferAsPPM(const std::string& filename, renderApi::Buffer& buffer, uint32_t width, uint32_t height);

#endif
