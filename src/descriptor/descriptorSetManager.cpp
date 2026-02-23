#include "descriptorSetManager.hpp"

#include "buffer/buffer.hpp"
#include "image/image.hpp"
#include "renderDevice.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <utility>
#include <vulkan/vulkan_core.h>

using namespace renderApi::descriptor;

DescriptorSet::DescriptorSet()
	: gpu_(nullptr), descriptorSet_(VK_NULL_HANDLE), layout_(VK_NULL_HANDLE) {}

DescriptorSet::~DescriptorSet() {
	destroy();
}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
	: gpu_(other.gpu_), descriptorSet_(other.descriptorSet_), layout_(other.layout_), bindings_(std::move(other.bindings_)) {
	other.descriptorSet_ = VK_NULL_HANDLE;
	other.layout_ = VK_NULL_HANDLE;
}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
	if (this != &other) {
		destroy();
		gpu_ = other.gpu_;
		descriptorSet_ = other.descriptorSet_;
		layout_ = other.layout_;
		bindings_ = std::move(other.bindings_);
		other.descriptorSet_ = VK_NULL_HANDLE;
		other.layout_ = VK_NULL_HANDLE;
	}
	return *this;
}

void DescriptorSet::addBinding(const DescriptorBinding& binding) {
	bindings_.push_back(binding);
}

void DescriptorSet::addBuffer(uint32_t binding, renderApi::Buffer* buffer, DescriptorType type, VkShaderStageFlags stages) {
	DescriptorBinding desc{};
	desc.binding = binding;
	desc.type = type;
	desc.count = 1;
	desc.stageFlags = stages;
	desc.buffer = buffer;
	bindings_.push_back(desc);
}

void DescriptorSet::addTexture(uint32_t binding, renderApi::Texture* texture, VkShaderStageFlags stages) {
	DescriptorBinding desc{};
	desc.binding = binding;
	desc.type = DescriptorType::COMBINED_IMAGE_SAMPLER;
	desc.count = 1;
	desc.stageFlags = stages;
	desc.texture = texture;
	bindings_.push_back(desc);
}

void DescriptorSet::addImage(uint32_t binding, renderApi::Image* image, DescriptorType type, VkShaderStageFlags stages) {
	DescriptorBinding desc{};
	desc.binding = binding;
	desc.type = type;
	desc.count = 1;
	desc.stageFlags = stages;
	desc.image = image;
	bindings_.push_back(desc);
}

void DescriptorSet::addSampler(uint32_t binding, renderApi::Sampler* sampler, VkShaderStageFlags stages) {
	DescriptorBinding desc{};
	desc.binding = binding;
	desc.type = DescriptorType::SAMPLER;
	desc.count = 1;
	desc.stageFlags = stages;
	desc.sampler = sampler;
	bindings_.push_back(desc);
}

VkDescriptorType DescriptorSet::convertDescriptorType(DescriptorType type) const {
	switch (type) {
	case DescriptorType::UNIFORM_BUFFER:
		return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	case DescriptorType::STORAGE_BUFFER:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	case DescriptorType::COMBINED_IMAGE_SAMPLER:
		return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	case DescriptorType::SAMPLED_IMAGE:
		return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	case DescriptorType::STORAGE_IMAGE:
		return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	case DescriptorType::SAMPLER:
		return VK_DESCRIPTOR_TYPE_SAMPLER;
	}
	return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}

bool DescriptorSet::build(renderApi::device::GPU* gpu, VkDescriptorPool pool) {
	if (bindings_.empty()) {
		std::cerr << "Cannot build descriptor set: no bindings" << std::endl;
		return false;
	}

	gpu_ = gpu;

	// Create layout
	std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
	layoutBindings.reserve(bindings_.size());

	for (const auto& binding : bindings_) {
		VkDescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.binding = binding.binding;
		layoutBinding.descriptorType = convertDescriptorType(binding.type);
		layoutBinding.descriptorCount = binding.count;
		layoutBinding.stageFlags = binding.stageFlags;
		layoutBinding.pImmutableSamplers = nullptr;
		layoutBindings.push_back(layoutBinding);
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
	layoutInfo.pBindings = layoutBindings.data();

	if (vkCreateDescriptorSetLayout(gpu_->device, &layoutInfo, nullptr, &layout_) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout" << std::endl;
		return false;
	}

	// Allocate descriptor set
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout_;

	if (vkAllocateDescriptorSets(gpu_->device, &allocInfo, &descriptorSet_) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set" << std::endl;
		vkDestroyDescriptorSetLayout(gpu_->device, layout_, nullptr);
		layout_ = VK_NULL_HANDLE;
		return false;
	}

	// Update descriptors
	update();

	return true;
}

void DescriptorSet::update() {
	if (!gpu_ || descriptorSet_ == VK_NULL_HANDLE) {
		return;
	}

	std::vector<VkWriteDescriptorSet> writes;
	std::vector<VkDescriptorBufferInfo> bufferInfos;
	std::vector<VkDescriptorImageInfo> imageInfos;

	writes.reserve(bindings_.size());
	bufferInfos.reserve(bindings_.size());
	imageInfos.reserve(bindings_.size());

	for (const auto& binding : bindings_) {
		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = descriptorSet_;
		write.dstBinding = binding.binding;
		write.dstArrayElement = 0;
		write.descriptorType = convertDescriptorType(binding.type);
		write.descriptorCount = binding.count;

		if (binding.buffer) {
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = binding.buffer->getHandle();
			bufferInfo.offset = 0;
			bufferInfo.range = binding.buffer->getSize();
			bufferInfos.push_back(bufferInfo);
			write.pBufferInfo = &bufferInfos.back();
		} else if (binding.texture) {
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = binding.texture->getImageView();
			imageInfo.sampler = binding.texture->getSamplerHandle();
			imageInfos.push_back(imageInfo);
			write.pImageInfo = &imageInfos.back();
		} else if (binding.image) {
			VkDescriptorImageInfo imageInfo{};
			if (binding.type == DescriptorType::STORAGE_IMAGE) {
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			} else {
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}
			imageInfo.imageView = binding.image->getView();
			imageInfo.sampler = VK_NULL_HANDLE;
			imageInfos.push_back(imageInfo);
			write.pImageInfo = &imageInfos.back();
		} else if (binding.sampler) {
			VkDescriptorImageInfo imageInfo{};
			imageInfo.sampler = binding.sampler->getHandle();
			imageInfo.imageView = VK_NULL_HANDLE;
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfos.push_back(imageInfo);
			write.pImageInfo = &imageInfos.back();
		}

		writes.push_back(write);
	}

	vkUpdateDescriptorSets(gpu_->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void DescriptorSet::destroy() {
	if (!gpu_ || !gpu_->device) {
		return;
	}

	// Note: descriptor set is freed when pool is destroyed
	descriptorSet_ = VK_NULL_HANDLE;

	if (layout_ != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(gpu_->device, layout_, nullptr);
		layout_ = VK_NULL_HANDLE;
	}

	bindings_.clear();
}

// ============================================================================
// DescriptorSetManager Implementation
// ============================================================================

DescriptorSetManager::DescriptorSetManager()
	: gpu_(nullptr), pool_(VK_NULL_HANDLE) {}

DescriptorSetManager::~DescriptorSetManager() {
	destroy();
}

DescriptorSet* DescriptorSetManager::createSet(uint32_t setIndex) {
	// Check if set already exists at this index
	for (size_t i = 0; i < setIndices_.size(); ++i) {
		if (setIndices_[i] == setIndex) {
			std::cerr << "Descriptor set at index " << setIndex << " already exists" << std::endl;
			return &sets_[i];
		}
	}

	// Insert at correct position to maintain sorted order
	auto it = std::lower_bound(setIndices_.begin(), setIndices_.end(), setIndex);
	size_t position = std::distance(setIndices_.begin(), it);

	setIndices_.insert(it, setIndex);
	sets_.insert(sets_.begin() + position, DescriptorSet());

	return &sets_[position];
}

DescriptorSet* DescriptorSetManager::getSet(uint32_t setIndex) {
	for (size_t i = 0; i < setIndices_.size(); ++i) {
		if (setIndices_[i] == setIndex) {
			return &sets_[i];
		}
	}
	return nullptr;
}

void DescriptorSetManager::removeSet(uint32_t setIndex) {
	for (size_t i = 0; i < setIndices_.size(); ++i) {
		if (setIndices_[i] == setIndex) {
			sets_[i].destroy();
			sets_.erase(sets_.begin() + i);
			setIndices_.erase(setIndices_.begin() + i);
			return;
		}
	}
}

void DescriptorSetManager::clearSets() {
	for (auto& set : sets_) {
		set.destroy();
	}
	sets_.clear();
	setIndices_.clear();
}

bool DescriptorSetManager::createPool() {
	if (!gpu_ || !gpu_->device) {
		std::cerr << "GPU not initialized" << std::endl;
		return false;
	}

	// Count descriptor types
	std::map<VkDescriptorType, uint32_t> typeCounts;

	for (const auto& set : sets_) {
		for (const auto& binding : set.getBindings()) {
			VkDescriptorType vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			switch (binding.type) {
			case DescriptorType::UNIFORM_BUFFER:
				vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				break;
			case DescriptorType::STORAGE_BUFFER:
				vkType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				break;
			case DescriptorType::COMBINED_IMAGE_SAMPLER:
				vkType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				break;
			case DescriptorType::SAMPLED_IMAGE:
				vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
				break;
			case DescriptorType::STORAGE_IMAGE:
				vkType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
				break;
			case DescriptorType::SAMPLER:
				vkType = VK_DESCRIPTOR_TYPE_SAMPLER;
				break;
			}
			typeCounts[vkType] += binding.count;
		}
	}

	// Create pool sizes
	std::vector<VkDescriptorPoolSize> poolSizes;
	poolSizes.reserve(typeCounts.size());

	for (const auto& [type, count] : typeCounts) {
		VkDescriptorPoolSize poolSize{};
		poolSize.type = type;
		poolSize.descriptorCount = count;
		poolSizes.push_back(poolSize);
	}

	if (poolSizes.empty()) {
		std::cerr << "No descriptors to allocate" << std::endl;
		return false;
	}

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = static_cast<uint32_t>(sets_.size());
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	if (vkCreateDescriptorPool(gpu_->device, &poolInfo, nullptr, &pool_) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool" << std::endl;
		return false;
	}

	return true;
}

bool DescriptorSetManager::build(renderApi::device::GPU* gpu) {
	if (sets_.empty()) {
		std::cerr << "No descriptor sets to build" << std::endl;
		return false;
	}

	gpu_ = gpu;

	// Create pool
	if (!createPool()) {
		return false;
	}

	// Build each set
	for (auto& set : sets_) {
		if (!set.build(gpu_, pool_)) {
			std::cerr << "Failed to build descriptor set" << std::endl;
			destroy();
			return false;
		}
	}

	return true;
}

void DescriptorSetManager::destroy() {
	if (!gpu_ || !gpu_->device) {
		return;
	}

	// Destroy all sets first (clears layouts)
	for (auto& set : sets_) {
		set.destroy();
	}

	// Destroy pool (automatically frees all descriptor sets)
	if (pool_ != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(gpu_->device, pool_, nullptr);
		pool_ = VK_NULL_HANDLE;
	}
}

std::vector<VkDescriptorSetLayout> DescriptorSetManager::getLayouts() const {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.reserve(sets_.size());

	for (const auto& set : sets_) {
		layouts.push_back(set.getLayout());
	}

	return layouts;
}

std::vector<VkDescriptorSet> DescriptorSetManager::getDescriptorSets() const {
	std::vector<VkDescriptorSet> descriptorSets;
	descriptorSets.reserve(sets_.size());

	for (const auto& set : sets_) {
		descriptorSets.push_back(set.getHandle());
	}

	return descriptorSets;
}
