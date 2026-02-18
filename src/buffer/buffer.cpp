#include "buffer.hpp"

#include "../device/renderDevice.hpp"

#include <cstring>
#include <iostream>
#include <vulkan/vulkan_core.h>

using namespace renderApi;

Buffer::Buffer()
	: gpu_(nullptr), buffer_(VK_NULL_HANDLE), memory_(VK_NULL_HANDLE), deviceAddress_(0), size_(0), type_(BufferType::VERTEX),
	  usage_(BufferUsage::STATIC), mappedPtr_(nullptr), persistentlyMapped_(false) {}

Buffer::~Buffer() { destroy(); }

Buffer::Buffer(Buffer&& other) noexcept
	: gpu_(other.gpu_), buffer_(other.buffer_), memory_(other.memory_), deviceAddress_(other.deviceAddress_), size_(other.size_),
	  type_(other.type_), usage_(other.usage_), mappedPtr_(other.mappedPtr_), persistentlyMapped_(other.persistentlyMapped_) {
	other.buffer_	 = VK_NULL_HANDLE;
	other.memory_	 = VK_NULL_HANDLE;
	other.mappedPtr_ = nullptr;
	other.size_		 = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
	if (this != &other) {
		destroy();
		gpu_				= other.gpu_;
		buffer_				= other.buffer_;
		memory_				= other.memory_;
		deviceAddress_		= other.deviceAddress_;
		size_				= other.size_;
		type_				= other.type_;
		usage_				= other.usage_;
		mappedPtr_			= other.mappedPtr_;
		persistentlyMapped_ = other.persistentlyMapped_;
		other.buffer_		= VK_NULL_HANDLE;
		other.memory_		= VK_NULL_HANDLE;
		other.mappedPtr_	= nullptr;
		other.size_			= 0;
	}
	return *this;
}

VkBufferUsageFlags Buffer::getVkUsageFlags() const {
	VkBufferUsageFlags flags = 0;

	switch (type_) {
	case BufferType::VERTEX:
		flags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		break;
	case BufferType::INDEX:
		flags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		break;
	case BufferType::UNIFORM:
		flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		break;
	case BufferType::STORAGE:
		flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		break;
	case BufferType::STAGING:
		flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		break;
	case BufferType::TRANSFER_SRC:
		flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		break;
	case BufferType::TRANSFER_DST:
		flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		break;
	}

	if (type_ == BufferType::STORAGE || type_ == BufferType::VERTEX || type_ == BufferType::INDEX) {
		flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	}

	return flags;
}

VkMemoryPropertyFlags Buffer::getMemoryFlags() const {
	switch (usage_) {
	case BufferUsage::STATIC:
		return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	case BufferUsage::DYNAMIC:
	case BufferUsage::STREAM:
		return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	}
	return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

bool Buffer::create(device::GPU* gpu, size_t size, BufferType type, BufferUsage usage) {
	destroy();

	gpu_	 = gpu;
	size_	 = size;
	type_	 = type;
	usage_	 = usage;

	if (size == 0) {
		std::cerr << "Cannot create buffer with size 0" << std::endl;
		return false;
	}

	VkDevice	 vkDevice = gpu->device;

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType	   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size		   = size;
	bufferInfo.usage	   = getVkUsageFlags();
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &buffer_) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer" << std::endl;
		return false;
	}

	// Get memory requirements
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(vkDevice, buffer_, &memRequirements);

	// Allocate memory
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize  = memRequirements.size;
	allocInfo.memoryTypeIndex = device::findMemoryType(gpu->physicalDevice, memRequirements.memoryTypeBits, getMemoryFlags());

	// Add device address allocation flag if needed
	VkMemoryAllocateFlagsInfo flagsInfo{};
	if (getVkUsageFlags() & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
		flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
		flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
		allocInfo.pNext = &flagsInfo;
	}

	if (vkAllocateMemory(vkDevice, &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
		std::cerr << "Failed to allocate buffer memory" << std::endl;
		vkDestroyBuffer(vkDevice, buffer_, nullptr);
		buffer_ = VK_NULL_HANDLE;
		return false;
	}

	// Bind memory
	vkBindBufferMemory(vkDevice, buffer_, memory_, 0);

	// Get device address if supported
	if (getVkUsageFlags() & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
		VkBufferDeviceAddressInfo addressInfo{};
		addressInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		addressInfo.buffer = buffer_;
		deviceAddress_	   = vkGetBufferDeviceAddress(vkDevice, &addressInfo);
	}

	// Persistent mapping for dynamic/stream buffers
	if (usage_ == BufferUsage::DYNAMIC || usage_ == BufferUsage::STREAM) {
		map();
		persistentlyMapped_ = true;
	}

	return true;
}

void Buffer::destroy() {
	if (!gpu_ || buffer_ == VK_NULL_HANDLE) return;

	VkDevice vkDevice = gpu_->device;

	if (persistentlyMapped_ && mappedPtr_) {
		unmap();
	}

	if (memory_ != VK_NULL_HANDLE) {
		vkFreeMemory(vkDevice, memory_, nullptr);
		memory_ = VK_NULL_HANDLE;
	}

	if (buffer_ != VK_NULL_HANDLE) {
		vkDestroyBuffer(vkDevice, buffer_, nullptr);
		buffer_ = VK_NULL_HANDLE;
	}

	size_				= 0;
	mappedPtr_			= nullptr;
	persistentlyMapped_ = false;
}

bool Buffer::resize(size_t newSize) {
	if (!isValid()) return false;

	BufferType	oldType	   = type_;
	BufferUsage oldUsage   = usage_;
	device::GPU* oldGpu = gpu_;

	destroy();
	return create(oldGpu, newSize, oldType, oldUsage);
}

bool Buffer::upload(const void* data, size_t size, size_t offset) {
	if (!isValid() || !data) return false;
	if (offset + size > size_) {
		std::cerr << "Upload size exceeds buffer size" << std::endl;
		return false;
	}

	// For host-visible buffers, just memcpy
	if (usage_ == BufferUsage::DYNAMIC || usage_ == BufferUsage::STREAM || type_ == BufferType::STAGING) {
		void* dst = map();
		if (!dst) return false;
		memcpy(static_cast<char*>(dst) + offset, data, size);
		if (!persistentlyMapped_) unmap();
		return true;
	}

	// For device-local buffers, use staging
	Buffer staging;
	if (!staging.create(gpu_, size, BufferType::STAGING, BufferUsage::STREAM)) {
		return false;
	}

	void* dst = staging.map();
	memcpy(dst, data, size);
	staging.unmap();

	// Copy buffer
	VkCommandBuffer cmd = gpu_->beginOneTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = offset;
	copyRegion.size		 = size;
	vkCmdCopyBuffer(cmd, staging.getHandle(), buffer_, 1, &copyRegion);

	gpu_->endOneTimeCommands(cmd);

	return true;
}

bool Buffer::download(void* data, size_t size, size_t offset) {
	if (!isValid() || !data) return false;
	if (offset + size > size_) {
		std::cerr << "Download size exceeds buffer size" << std::endl;
		return false;
	}

	// For host-visible buffers, just memcpy
	if (usage_ == BufferUsage::DYNAMIC || usage_ == BufferUsage::STREAM || type_ == BufferType::STAGING) {
		void* src = map();
		if (!src) return false;
		memcpy(data, static_cast<char*>(src) + offset, size);
		if (!persistentlyMapped_) unmap();
		return true;
	}

	// For device-local buffers, use staging
	Buffer staging;
	if (!staging.create(gpu_, size, BufferType::STAGING, BufferUsage::STREAM)) {
		return false;
	}

	// Copy to staging
	VkCommandBuffer cmd = gpu_->beginOneTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = offset;
	copyRegion.dstOffset = 0;
	copyRegion.size		 = size;
	vkCmdCopyBuffer(cmd, buffer_, staging.getHandle(), 1, &copyRegion);

	gpu_->endOneTimeCommands(cmd);

	void* src = staging.map();
	memcpy(data, src, size);
	staging.unmap();

	return true;
}

void* Buffer::map() {
	if (!isValid()) return nullptr;
	if (mappedPtr_) return mappedPtr_;

	VkDevice vkDevice = gpu_->device;
	if (vkMapMemory(vkDevice, memory_, 0, size_, 0, &mappedPtr_) != VK_SUCCESS) {
		std::cerr << "Failed to map buffer memory" << std::endl;
		return nullptr;
	}
	return mappedPtr_;
}

void Buffer::unmap() {
	if (!isValid() || !mappedPtr_) return;

	VkDevice vkDevice = gpu_->device;
	vkUnmapMemory(vkDevice, memory_);
	mappedPtr_ = nullptr;
}

VkDeviceAddress Buffer::getDeviceAddress() const { return deviceAddress_; }
