#include "descriptors.hpp"

#include "gpuContext.hpp"

#include <iostream>
#include <stdexcept>

namespace renderApi {

	// ==================== DescriptorLayout ====================

	DescriptorLayout::DescriptorLayout() : context_(nullptr), layout_(VK_NULL_HANDLE) {}

	DescriptorLayout::~DescriptorLayout() { destroy(); }

	DescriptorLayout::DescriptorLayout(DescriptorLayout&& other) noexcept
		: context_(other.context_), layout_(other.layout_), bindings_(std::move(other.bindings_)) {
		other.layout_ = VK_NULL_HANDLE;
	}

	DescriptorLayout& DescriptorLayout::operator=(DescriptorLayout&& other) noexcept {
		if (this != &other) {
			destroy();
			context_	  = other.context_;
			layout_		  = other.layout_;
			bindings_	  = std::move(other.bindings_);
			other.layout_ = VK_NULL_HANDLE;
		}
		return *this;
	}

	DescriptorLayout& DescriptorLayout::addUniformBuffer(uint32_t binding, VkShaderStageFlags stage) {
		VkDescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.binding			 = binding;
		layoutBinding.descriptorType	 = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBinding.descriptorCount	 = 1;
		layoutBinding.stageFlags		 = stage;
		layoutBinding.pImmutableSamplers = nullptr;
		bindings_.push_back(layoutBinding);
		return *this;
	}

	DescriptorLayout& DescriptorLayout::addStorageBuffer(uint32_t binding, VkShaderStageFlags stage) {
		VkDescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.binding			 = binding;
		layoutBinding.descriptorType	 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBinding.descriptorCount	 = 1;
		layoutBinding.stageFlags		 = stage;
		layoutBinding.pImmutableSamplers = nullptr;
		bindings_.push_back(layoutBinding);
		return *this;
	}

	DescriptorLayout& DescriptorLayout::addCombinedImageSampler(uint32_t binding, VkShaderStageFlags stage) {
		VkDescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.binding			 = binding;
		layoutBinding.descriptorType	 = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		layoutBinding.descriptorCount	 = 1;
		layoutBinding.stageFlags		 = stage;
		layoutBinding.pImmutableSamplers = nullptr;
		bindings_.push_back(layoutBinding);
		return *this;
	}

	bool DescriptorLayout::build(GPUContext& context) {
		context_ = &context;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType		= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings_.size());
		layoutInfo.pBindings	= bindings_.data();

		if (vkCreateDescriptorSetLayout(context_->getDevice(), &layoutInfo, nullptr, &layout_) != VK_SUCCESS) {
			std::cerr << "Failed to create descriptor set layout!" << std::endl;
			return false;
		}
		return true;
	}

	void DescriptorLayout::destroy() {
		if (layout_ != VK_NULL_HANDLE && context_) {
			vkDestroyDescriptorSetLayout(context_->getDevice(), layout_, nullptr);
			layout_ = VK_NULL_HANDLE;
		}
	}

	// ==================== DescriptorSet ====================

	DescriptorSet::DescriptorSet() : context_(nullptr), set_(VK_NULL_HANDLE), layout_(nullptr) {}

	DescriptorSet::~DescriptorSet() { free(); }

	DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept : context_(other.context_), set_(other.set_), layout_(other.layout_) {
		other.set_	  = VK_NULL_HANDLE;
		other.layout_ = nullptr;
	}

	DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
		if (this != &other) {
			free();
			context_	  = other.context_;
			set_		  = other.set_;
			layout_		  = other.layout_;
			other.set_	  = VK_NULL_HANDLE;
			other.layout_ = nullptr;
		}
		return *this;
	}

	bool DescriptorSet::allocate(GPUContext& context, const DescriptorLayout& layout) {
		context_ = &context;
		layout_	 = &layout;

		VkDescriptorSetLayout layoutHandle = layout.getHandle();

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType				 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool	 = context_->getDescriptorPool();
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts		 = &layoutHandle;

		if (vkAllocateDescriptorSets(context_->getDevice(), &allocInfo, &set_) != VK_SUCCESS) {
			std::cerr << "Failed to allocate descriptor set!" << std::endl;
			return false;
		}
		return true;
	}

	void DescriptorSet::updateUniformBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = buffer;
		bufferInfo.offset = offset;
		bufferInfo.range  = range;

		VkWriteDescriptorSet descriptorWrite{};
		descriptorWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet			= set_;
		descriptorWrite.dstBinding		= binding;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType	= VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo		= &bufferInfo;

		vkUpdateDescriptorSets(context_->getDevice(), 1, &descriptorWrite, 0, nullptr);
	}

	void DescriptorSet::updateStorageBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = buffer;
		bufferInfo.offset = offset;
		bufferInfo.range  = range;

		VkWriteDescriptorSet descriptorWrite{};
		descriptorWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet			= set_;
		descriptorWrite.dstBinding		= binding;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType	= VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo		= &bufferInfo;

		vkUpdateDescriptorSets(context_->getDevice(), 1, &descriptorWrite, 0, nullptr);
	}

	void DescriptorSet::free() {
		if (set_ != VK_NULL_HANDLE && context_) {
			vkFreeDescriptorSets(context_->getDevice(), context_->getDescriptorPool(), 1, &set_);
			set_ = VK_NULL_HANDLE;
		}
	}

	// ==================== DescriptorPool ====================

	DescriptorPool::DescriptorPool() : context_(nullptr), pool_(VK_NULL_HANDLE) {}

	DescriptorPool::~DescriptorPool() { destroy(); }

	DescriptorPool::DescriptorPool(DescriptorPool&& other) noexcept : context_(other.context_), pool_(other.pool_) { other.pool_ = VK_NULL_HANDLE; }

	DescriptorPool& DescriptorPool::operator=(DescriptorPool&& other) noexcept {
		if (this != &other) {
			destroy();
			context_	= other.context_;
			pool_		= other.pool_;
			other.pool_ = VK_NULL_HANDLE;
		}
		return *this;
	}

	bool
	DescriptorPool::create(GPUContext& context, uint32_t maxSets, uint32_t uniformBuffers, uint32_t storageBuffers, uint32_t combinedImageSamplers) {
		context_ = &context;

		std::vector<VkDescriptorPoolSize> poolSizes;

		if (uniformBuffers > 0) {
			poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBuffers});
		}
		if (storageBuffers > 0) {
			poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storageBuffers});
		}
		if (combinedImageSamplers > 0) {
			poolSizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, combinedImageSamplers});
		}

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes	   = poolSizes.data();
		poolInfo.maxSets	   = maxSets;
		poolInfo.flags		   = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

		if (vkCreateDescriptorPool(context_->getDevice(), &poolInfo, nullptr, &pool_) != VK_SUCCESS) {
			std::cerr << "Failed to create descriptor pool!" << std::endl;
			return false;
		}
		return true;
	}

	void DescriptorPool::destroy() {
		if (pool_ != VK_NULL_HANDLE && context_) {
			vkDestroyDescriptorPool(context_->getDevice(), pool_, nullptr);
			pool_ = VK_NULL_HANDLE;
		}
	}

} // namespace renderApi
