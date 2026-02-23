#include "image.hpp"

#include "buffer/buffer.hpp"
#include "renderDevice.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vulkan/vulkan_core.h>

using namespace renderApi;

// ============================================================================
// Image Implementation
// ============================================================================

Image::Image()
	: gpu_(nullptr), image_(VK_NULL_HANDLE), imageView_(VK_NULL_HANDLE), memory_(VK_NULL_HANDLE), format_(VK_FORMAT_UNDEFINED), width_(0),
	  height_(0), depth_(0), mipLevels_(1), arrayLayers_(1), type_(ImageType::IMAGE_2D), usage_(ImageUsage::TEXTURE),
	  currentLayout_(ImageLayout::UNDEFINED), aspectMask_(VK_IMAGE_ASPECT_COLOR_BIT) {}

Image::~Image() { destroy(); }

Image::Image(Image&& other) noexcept
	: gpu_(other.gpu_), image_(other.image_), imageView_(other.imageView_), memory_(other.memory_), format_(other.format_), width_(other.width_),
	  height_(other.height_), depth_(other.depth_), mipLevels_(other.mipLevels_), arrayLayers_(other.arrayLayers_), type_(other.type_),
	  usage_(other.usage_), currentLayout_(other.currentLayout_), aspectMask_(other.aspectMask_) {
	other.image_	 = VK_NULL_HANDLE;
	other.imageView_ = VK_NULL_HANDLE;
	other.memory_	 = VK_NULL_HANDLE;
}

Image& Image::operator=(Image&& other) noexcept {
	if (this != &other) {
		destroy();
		gpu_		   = other.gpu_;
		image_		   = other.image_;
		imageView_	   = other.imageView_;
		memory_		   = other.memory_;
		format_		   = other.format_;
		width_		   = other.width_;
		height_		   = other.height_;
		depth_		   = other.depth_;
		mipLevels_	   = other.mipLevels_;
		arrayLayers_   = other.arrayLayers_;
		type_		   = other.type_;
		usage_		   = other.usage_;
		currentLayout_ = other.currentLayout_;
		aspectMask_	   = other.aspectMask_;

		other.image_	 = VK_NULL_HANDLE;
		other.imageView_ = VK_NULL_HANDLE;
		other.memory_	 = VK_NULL_HANDLE;
	}
	return *this;
}

VkImageUsageFlags Image::getVkUsageFlags() const {
	VkImageUsageFlags flags = 0;

	switch (usage_) {
	case ImageUsage::TEXTURE:
		flags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		break;
	case ImageUsage::RENDER_TARGET:
		flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		break;
	case ImageUsage::DEPTH_STENCIL:
		flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		break;
	case ImageUsage::STORAGE:
		flags = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		break;
	case ImageUsage::TRANSFER_SRC:
		flags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		break;
	case ImageUsage::TRANSFER_DST:
		flags = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		break;
	case ImageUsage::COMBINED:
		flags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
				VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		break;
	}

	return flags;
}

VkImageAspectFlags Image::getAspectMask() const {
	if (format_ == VK_FORMAT_D32_SFLOAT || format_ == VK_FORMAT_D32_SFLOAT_S8_UINT || format_ == VK_FORMAT_D24_UNORM_S8_UINT ||
		format_ == VK_FORMAT_D16_UNORM) {
		return VK_IMAGE_ASPECT_DEPTH_BIT;
	}
	return VK_IMAGE_ASPECT_COLOR_BIT;
}

VkImageLayout Image::convertLayout(ImageLayout layout) const {
	switch (layout) {
	case ImageLayout::UNDEFINED:
		return VK_IMAGE_LAYOUT_UNDEFINED;
	case ImageLayout::GENERAL:
		return VK_IMAGE_LAYOUT_GENERAL;
	case ImageLayout::COLOR_ATTACHMENT:
		return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	case ImageLayout::DEPTH_STENCIL_ATTACHMENT:
		return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	case ImageLayout::SHADER_READ_ONLY:
		return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	case ImageLayout::TRANSFER_SRC:
		return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	case ImageLayout::TRANSFER_DST:
		return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	case ImageLayout::PRESENT_SRC:
		return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	}
	return VK_IMAGE_LAYOUT_UNDEFINED;
}

bool Image::create(device::GPU* gpu, const ImageCreateInfo& info) {
	destroy();

	gpu_		 = gpu;
	width_		 = info.width;
	height_		 = info.height;
	depth_		 = info.depth;
	format_		 = info.format;
	mipLevels_	 = info.mipLevels;
	arrayLayers_ = info.arrayLayers;
	type_		 = info.type;
	usage_		 = info.usage;
	aspectMask_	 = getAspectMask();

	if (!gpu_ || !gpu_->device) {
		std::cerr << "GPU not initialized" << std::endl;
		return false;
	}

	VkImageType imageType = VK_IMAGE_TYPE_2D;
	switch (type_) {
	case ImageType::IMAGE_1D:
		imageType = VK_IMAGE_TYPE_1D;
		break;
	case ImageType::IMAGE_2D:
	case ImageType::CUBE:
		imageType = VK_IMAGE_TYPE_2D;
		break;
	case ImageType::IMAGE_3D:
		imageType = VK_IMAGE_TYPE_3D;
		break;
	}

	VkImageCreateInfo imageInfo{};
	imageInfo.sType		   = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType	   = imageType;
	imageInfo.extent.width  = width_;
	imageInfo.extent.height = height_;
	imageInfo.extent.depth  = depth_;
	imageInfo.mipLevels	   = mipLevels_;
	imageInfo.arrayLayers   = arrayLayers_;
	imageInfo.format	   = format_;
	imageInfo.tiling	   = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage		   = getVkUsageFlags();
	imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples	   = info.samples;
	imageInfo.flags		   = (type_ == ImageType::CUBE) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;

	if (vkCreateImage(gpu_->device, &imageInfo, nullptr, &image_) != VK_SUCCESS) {
		std::cerr << "Failed to create image" << std::endl;
		return false;
	}

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(gpu_->device, image_, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType			 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex =
			device::findMemoryType(gpu_->physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	if (vkAllocateMemory(gpu_->device, &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
		std::cerr << "Failed to allocate image memory" << std::endl;
		vkDestroyImage(gpu_->device, image_, nullptr);
		image_ = VK_NULL_HANDLE;
		return false;
	}

	vkBindImageMemory(gpu_->device, image_, memory_, 0);

	VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
	switch (type_) {
	case ImageType::IMAGE_1D:
		viewType = arrayLayers_ > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
		break;
	case ImageType::IMAGE_2D:
		viewType = arrayLayers_ > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
		break;
	case ImageType::IMAGE_3D:
		viewType = VK_IMAGE_VIEW_TYPE_3D;
		break;
	case ImageType::CUBE:
		viewType = arrayLayers_ > 6 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
		break;
	}

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType							= VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image							= image_;
	viewInfo.viewType						= viewType;
	viewInfo.format							= format_;
	viewInfo.subresourceRange.aspectMask	= aspectMask_;
	viewInfo.subresourceRange.baseMipLevel	= 0;
	viewInfo.subresourceRange.levelCount	= mipLevels_;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount	= arrayLayers_;

	if (vkCreateImageView(gpu_->device, &viewInfo, nullptr, &imageView_) != VK_SUCCESS) {
		std::cerr << "Failed to create image view" << std::endl;
		destroy();
		return false;
	}

	currentLayout_ = ImageLayout::UNDEFINED;

	if (info.generateMipmaps && mipLevels_ > 1) {
		transitionLayout(ImageLayout::TRANSFER_DST);
	}

	return true;
}

void Image::destroy() {
	if (!gpu_ || !gpu_->device) return;

	if (imageView_ != VK_NULL_HANDLE) {
		vkDestroyImageView(gpu_->device, imageView_, nullptr);
		imageView_ = VK_NULL_HANDLE;
	}

	if (image_ != VK_NULL_HANDLE) {
		vkDestroyImage(gpu_->device, image_, nullptr);
		image_ = VK_NULL_HANDLE;
	}

	if (memory_ != VK_NULL_HANDLE) {
		vkFreeMemory(gpu_->device, memory_, nullptr);
		memory_ = VK_NULL_HANDLE;
	}
}

void Image::transitionLayout(VkCommandBuffer cmd, ImageLayout newLayout) {
	VkImageMemoryBarrier barrier{};
	barrier.sType							= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout						= convertLayout(currentLayout_);
	barrier.newLayout						= convertLayout(newLayout);
	barrier.srcQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.image							= image_;
	barrier.subresourceRange.aspectMask		= aspectMask_;
	barrier.subresourceRange.baseMipLevel	= 0;
	barrier.subresourceRange.levelCount		= mipLevels_;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount		= arrayLayers_;

	VkPipelineStageFlags srcStage;
	VkPipelineStageFlags dstStage;

	if (currentLayout_ == ImageLayout::UNDEFINED && newLayout == ImageLayout::TRANSFER_DST) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		srcStage			  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStage			  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (currentLayout_ == ImageLayout::TRANSFER_DST && newLayout == ImageLayout::SHADER_READ_ONLY) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		srcStage			  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dstStage			  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else if (currentLayout_ == ImageLayout::UNDEFINED && newLayout == ImageLayout::DEPTH_STENCIL_ATTACHMENT) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		srcStage			  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStage			  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	} else if (currentLayout_ == ImageLayout::UNDEFINED && newLayout == ImageLayout::COLOR_ATTACHMENT) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		srcStage			  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStage			  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	} else if (currentLayout_ == ImageLayout::COLOR_ATTACHMENT && newLayout == ImageLayout::SHADER_READ_ONLY) {
		barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		srcStage			  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dstStage			  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else if (currentLayout_ == ImageLayout::COLOR_ATTACHMENT && newLayout == ImageLayout::TRANSFER_SRC) {
		barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		srcStage			  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dstStage			  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (currentLayout_ == ImageLayout::TRANSFER_SRC && newLayout == ImageLayout::SHADER_READ_ONLY) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		srcStage			  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dstStage			  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else if (currentLayout_ == ImageLayout::SHADER_READ_ONLY && newLayout == ImageLayout::TRANSFER_DST) {
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		srcStage			  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dstStage			  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (currentLayout_ == ImageLayout::COLOR_ATTACHMENT && newLayout == ImageLayout::PRESENT_SRC) {
		barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		srcStage			  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dstStage			  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	} else {
		// General transition
		barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
		srcStage			  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		dstStage			  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	}

	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	currentLayout_ = newLayout;
}

void Image::transitionLayout(ImageLayout newLayout) {
	VkCommandBuffer cmd = gpu_->beginOneTimeCommands();
	transitionLayout(cmd, newLayout);
	gpu_->endOneTimeCommands(cmd);
}

bool Image::uploadData(const void* data, size_t size) {
	return uploadDataStaged(data, size);
}

bool Image::uploadDataStaged(const void* data, size_t size) {
	if (!isValid() || !data) return false;

	Buffer staging;
	if (!staging.create(gpu_, size, BufferType::STAGING, BufferUsage::STREAM)) {
		return false;
	}

	void* mapped = staging.map();
	memcpy(mapped, data, size);
	staging.unmap();

	VkCommandBuffer cmd = gpu_->beginOneTimeCommands();

	transitionLayout(cmd, ImageLayout::TRANSFER_DST);

	VkBufferImageCopy region{};
	region.bufferOffset						= 0;
	region.bufferRowLength					= 0;
	region.bufferImageHeight				= 0;
	region.imageSubresource.aspectMask		= aspectMask_;
	region.imageSubresource.mipLevel		= 0;
	region.imageSubresource.baseArrayLayer	= 0;
	region.imageSubresource.layerCount		= arrayLayers_;
	region.imageOffset						= {0, 0, 0};
	region.imageExtent						= {width_, height_, depth_};

	vkCmdCopyBufferToImage(cmd, staging.getHandle(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	transitionLayout(cmd, ImageLayout::SHADER_READ_ONLY);

	gpu_->endOneTimeCommands(cmd);

	return true;
}

void Image::generateMipmaps() {
	if (mipLevels_ <= 1) return;

	VkCommandBuffer cmd = gpu_->beginOneTimeCommands();

	VkImageMemoryBarrier barrier{};
	barrier.sType							= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.image							= image_;
	barrier.srcQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.subresourceRange.aspectMask		= VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount		= arrayLayers_;
	barrier.subresourceRange.levelCount		= 1;

	int32_t mipWidth  = width_;
	int32_t mipHeight = height_;

	for (uint32_t i = 1; i < mipLevels_; i++) {
		barrier.subresourceRange.baseMipLevel = i - 1;
		barrier.oldLayout					  = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout					  = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask				  = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask				  = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmd,
							 VK_PIPELINE_STAGE_TRANSFER_BIT,
							 VK_PIPELINE_STAGE_TRANSFER_BIT,
							 0,
							 0,
							 nullptr,
							 0,
							 nullptr,
							 1,
							 &barrier);

		VkImageBlit blit{};
		blit.srcOffsets[0]					 = {0, 0, 0};
		blit.srcOffsets[1]					 = {mipWidth, mipHeight, 1};
		blit.srcSubresource.aspectMask		 = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel		 = i - 1;
		blit.srcSubresource.baseArrayLayer	 = 0;
		blit.srcSubresource.layerCount		 = arrayLayers_;
		blit.dstOffsets[0]					 = {0, 0, 0};
		blit.dstOffsets[1]					 = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
		blit.dstSubresource.aspectMask		 = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel		 = i;
		blit.dstSubresource.baseArrayLayer	 = 0;
		blit.dstSubresource.layerCount		 = arrayLayers_;

		vkCmdBlitImage(cmd,
					   image_,
					   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					   image_,
					   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					   1,
					   &blit,
					   VK_FILTER_LINEAR);

		barrier.oldLayout	  = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout	  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd,
							 VK_PIPELINE_STAGE_TRANSFER_BIT,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							 0,
							 0,
							 nullptr,
							 0,
							 nullptr,
							 1,
							 &barrier);

		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;
	}

	barrier.subresourceRange.baseMipLevel = mipLevels_ - 1;
	barrier.oldLayout					  = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout					  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask				  = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask				  = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	gpu_->endOneTimeCommands(cmd);

	currentLayout_ = ImageLayout::SHADER_READ_ONLY;
}

void Image::copyToBuffer(Buffer& buffer) {
	VkCommandBuffer cmd = gpu_->beginOneTimeCommands();

	transitionLayout(cmd, ImageLayout::TRANSFER_SRC);

	VkBufferImageCopy region{};
	region.bufferOffset						= 0;
	region.bufferRowLength					= 0;
	region.bufferImageHeight				= 0;
	region.imageSubresource.aspectMask		= aspectMask_;
	region.imageSubresource.mipLevel		= 0;
	region.imageSubresource.baseArrayLayer	= 0;
	region.imageSubresource.layerCount		= arrayLayers_;
	region.imageOffset						= {0, 0, 0};
	region.imageExtent						= {width_, height_, depth_};

	vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.getHandle(), 1, &region);

	gpu_->endOneTimeCommands(cmd);
}

void Image::copyFromBuffer(const Buffer& buffer) {
	VkCommandBuffer cmd = gpu_->beginOneTimeCommands();

	transitionLayout(cmd, ImageLayout::TRANSFER_DST);

	VkBufferImageCopy region{};
	region.bufferOffset						= 0;
	region.bufferRowLength					= 0;
	region.bufferImageHeight				= 0;
	region.imageSubresource.aspectMask		= aspectMask_;
	region.imageSubresource.mipLevel		= 0;
	region.imageSubresource.baseArrayLayer	= 0;
	region.imageSubresource.layerCount		= arrayLayers_;
	region.imageOffset						= {0, 0, 0};
	region.imageExtent						= {width_, height_, depth_};

	vkCmdCopyBufferToImage(cmd, buffer.getHandle(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	transitionLayout(cmd, ImageLayout::SHADER_READ_ONLY);

	gpu_->endOneTimeCommands(cmd);
}

// ============================================================================
// Sampler Implementation
// ============================================================================

Sampler::Sampler() : gpu_(nullptr), sampler_(VK_NULL_HANDLE) {}

Sampler::~Sampler() { destroy(); }

Sampler::Sampler(Sampler&& other) noexcept : gpu_(other.gpu_), sampler_(other.sampler_) { other.sampler_ = VK_NULL_HANDLE; }

Sampler& Sampler::operator=(Sampler&& other) noexcept {
	if (this != &other) {
		destroy();
		gpu_			= other.gpu_;
		sampler_		= other.sampler_;
		other.sampler_	= VK_NULL_HANDLE;
	}
	return *this;
}

bool Sampler::create(device::GPU* gpu, const SamplerCreateInfo& info) {
	destroy();

	gpu_ = gpu;

	if (!gpu_ || !gpu_->device) {
		std::cerr << "GPU not initialized" << std::endl;
		return false;
	}

	VkFilter magFilter = (info.magFilter == FilterMode::LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
	VkFilter minFilter = (info.minFilter == FilterMode::LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

	auto convertAddressMode = [](AddressMode mode) -> VkSamplerAddressMode {
		switch (mode) {
		case AddressMode::REPEAT:
			return VK_SAMPLER_ADDRESS_MODE_REPEAT;
		case AddressMode::MIRRORED_REPEAT:
			return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
		case AddressMode::CLAMP_TO_EDGE:
			return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		case AddressMode::CLAMP_TO_BORDER:
			return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		}
		return VK_SAMPLER_ADDRESS_MODE_REPEAT;
	};

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType					= VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter				= magFilter;
	samplerInfo.minFilter				= minFilter;
	samplerInfo.addressModeU			= convertAddressMode(info.addressModeU);
	samplerInfo.addressModeV			= convertAddressMode(info.addressModeV);
	samplerInfo.addressModeW			= convertAddressMode(info.addressModeW);
	samplerInfo.anisotropyEnable		= info.enableAnisotropy ? VK_TRUE : VK_FALSE;
	samplerInfo.maxAnisotropy			= info.maxAnisotropy;
	samplerInfo.borderColor				= VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable			= VK_FALSE;
	samplerInfo.compareOp				= VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode				= VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias				= info.mipLodBias;
	samplerInfo.minLod					= info.minLod;
	samplerInfo.maxLod					= info.maxLod;

	if (vkCreateSampler(gpu_->device, &samplerInfo, nullptr, &sampler_) != VK_SUCCESS) {
		std::cerr << "Failed to create sampler" << std::endl;
		return false;
	}

	return true;
}

void Sampler::destroy() {
	if (!gpu_ || !gpu_->device) return;

	if (sampler_ != VK_NULL_HANDLE) {
		vkDestroySampler(gpu_->device, sampler_, nullptr);
		sampler_ = VK_NULL_HANDLE;
	}
}

// ============================================================================
// Texture Implementation
// ============================================================================

Texture::Texture() {}

Texture::~Texture() { destroy(); }

Texture::Texture(Texture&& other) noexcept : image_(std::move(other.image_)), sampler_(std::move(other.sampler_)) {}

Texture& Texture::operator=(Texture&& other) noexcept {
	if (this != &other) {
		destroy();
		image_	 = std::move(other.image_);
		sampler_ = std::move(other.sampler_);
	}
	return *this;
}

bool Texture::create(device::GPU* gpu, const ImageCreateInfo& imageInfo, const SamplerCreateInfo& samplerInfo) {
	destroy();

	if (!image_.create(gpu, imageInfo)) {
		return false;
	}

	if (!sampler_.create(gpu, samplerInfo)) {
		image_.destroy();
		return false;
	}

	return true;
}

bool Texture::createFromFile(device::GPU* gpu, const char* filename, bool generateMipmaps) {
	// TODO: Implement image loading from file (stb_image, etc.)
	std::cerr << "Texture::createFromFile not yet implemented" << std::endl;
	(void)gpu;
	(void)filename;
	(void)generateMipmaps;
	return false;
}

void Texture::destroy() {
	sampler_.destroy();
	image_.destroy();
}

bool Texture::uploadData(const void* data, size_t size) { return image_.uploadData(data, size); }
