#ifndef UTILS_HPP
# define UTILS_HPP

# include <cstddef>
# include <string>
# include <vulkan/vulkan_core.h>

std::string	   generateRandomString(size_t length = 8);
VkShaderModule createShaderModule(VkDevice device, const std::string& path);

#endif
