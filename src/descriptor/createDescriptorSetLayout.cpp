#include "buffer/buffer.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <vulkan/vulkan_core.h>

VkDescriptorType bufferTypeToDescriptorType(renderApi::BufferType type) {
	switch (type) {
	case renderApi::BufferType::UNIFORM:
		return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	case renderApi::BufferType::STORAGE:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	case renderApi::BufferType::VERTEX:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	case renderApi::BufferType::INDEX:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	default:
		throw std::runtime_error("Buffer type not supported for descriptor set layout");
	}
}

VkDescriptorSetLayout createDescriptorSetLayoutFromBuffers(VkDevice device, const std::vector<renderApi::Buffer*>& buffers, const std::vector<VkShaderStageFlags>& stages) {
	if (buffers.empty()) {
		throw std::runtime_error("Cannot create descriptor set layout from empty buffer list");
	}

	if (!stages.empty() && stages.size() != buffers.size()) {
		throw std::runtime_error("Stages vector must have the same size as buffers vector");
	}

	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.reserve(buffers.size());

	for (size_t i = 0; i < buffers.size(); ++i) {
		if (!buffers[i]) {
			throw std::runtime_error("Null buffer in buffer list at index " + std::to_string(i));
		}

		VkDescriptorSetLayoutBinding binding = {};
		binding.binding						 = static_cast<uint32_t>(i);
		binding.descriptorType				 = bufferTypeToDescriptorType(buffers[i]->getType());
		binding.descriptorCount				 = 1;
		binding.stageFlags					 = stages.empty() ? VK_SHADER_STAGE_ALL : stages[i];
		binding.pImmutableSamplers			 = nullptr;

		bindings.push_back(binding);
	}

	VkDescriptorSetLayoutCreateInfo createInfo = {};
	createInfo.sType						   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	createInfo.bindingCount					   = static_cast<uint32_t>(bindings.size());
	createInfo.pBindings					   = bindings.data();

	VkDescriptorSetLayout descriptorSetLayout;
	VkResult			  result = vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &descriptorSetLayout);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout from buffers!");
	}

	return descriptorSetLayout;
}

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
	VkDescriptorSetLayoutCreateInfo createInfo = {};
	createInfo.sType						   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	createInfo.bindingCount					   = static_cast<uint32_t>(bindings.size());
	createInfo.pBindings					   = bindings.data();

	VkDescriptorSetLayout descriptorSetLayout;
	VkResult			  result = vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &descriptorSetLayout);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout!");
	}

	return descriptorSetLayout;
}

void destroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout layout) {
	if (layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(device, layout, nullptr);
	}
}
