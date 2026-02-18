#ifndef CREATE_DESCRIPTOR_SET_LAYOUT_HPP
# define CREATE_DESCRIPTOR_SET_LAYOUT_HPP

# include <vector>
# include <vulkan/vulkan_core.h>

namespace renderApi {
	class Buffer;
}

VkDescriptorSetLayout	createDescriptorSetLayoutFromBuffers(VkDevice device, const std::vector<renderApi::Buffer*>& buffers, const std::vector<VkShaderStageFlags>& stages);
VkDescriptorSetLayout	createDescriptorSetLayout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings);
void					destroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout layout);

#endif
