#include "device/renderDevice.hpp"
#include "pipeline/graphicsPipeline.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace renderApi::gpuTask;

bool GraphicsPipeline::createSwapchain() {
	if (!gpu_ || !gpu_->device || surface_ == VK_NULL_HANDLE) {
		std::cerr << "GraphicsPipeline: Cannot create swapchain without surface" << std::endl;
		return false;
	}

	if (window_) {
		int windowWidth = 0;
		int windowHeight = 0;
		SDL_GetWindowSize(window_, &windowWidth, &windowHeight);
		if (windowWidth <= 0 || windowHeight <= 0) {
			std::cerr << "GraphicsPipeline: Window size invalid for swapchain" << std::endl;
			return false;
		}
	}

	VkSurfaceCapabilitiesKHR capabilities;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_->physicalDevice, surface_, &capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_->physicalDevice, surface_, &formatCount, nullptr);
	std::vector<VkSurfaceFormatKHR> formats(formatCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_->physicalDevice, surface_, &formatCount, formats.data());

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_->physicalDevice, surface_, &presentModeCount, nullptr);
	std::vector<VkPresentModeKHR> presentModes(presentModeCount);
	vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_->physicalDevice, surface_, &presentModeCount, presentModes.data());

	VkSurfaceFormatKHR surfaceFormat   = formats[0];
	VkFormat		   preferredFormat = !colorFormats_.empty() ? colorFormats_[0] : VK_FORMAT_B8G8R8A8_UNORM;

	for (const auto& format : formats) {
		if (format.format == preferredFormat && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			surfaceFormat = format;
			break;
		}
	}

	if (surfaceFormat.format != preferredFormat) {
		for (const auto& format : formats) {
			if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				surfaceFormat = format;
				if (!colorFormats_.empty()) {
					colorFormats_[0] = VK_FORMAT_B8G8R8A8_UNORM;
				}
				break;
			}
		}
	}

	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	for (const auto& mode : presentModes) {
		if (mode == preferredPresentMode_) {
			presentMode = preferredPresentMode_;
			break;
		}
	}

	const char* modeName = "UNKNOWN";
	switch (presentMode) {
	case VK_PRESENT_MODE_IMMEDIATE_KHR:
		modeName = "IMMEDIATE (No VSync)";
		break;
	case VK_PRESENT_MODE_MAILBOX_KHR:
		modeName = "MAILBOX (Triple buffering)";
		break;
	case VK_PRESENT_MODE_FIFO_KHR:
		modeName = "FIFO (VSync)";
		break;
	case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
		modeName = "FIFO_RELAXED";
		break;
	default:
		break;
	}
	std::cout << "GraphicsPipeline: Using present mode: " << modeName << std::endl;

	VkExtent2D extent;
	if (capabilities.currentExtent.width != UINT32_MAX) {
		extent = capabilities.currentExtent;
	} else {
		if (window_) {
			int windowWidth = 0;
			int windowHeight = 0;
			SDL_GetWindowSize(window_, &windowWidth, &windowHeight);
			width_ = static_cast<uint32_t>(windowWidth);
			height_ = static_cast<uint32_t>(windowHeight);
		}
		extent		  = {width_, height_};
		extent.width  = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
	}

	uint32_t imageCount;
	if (requestedImageCount_ > 0) {
		imageCount = requestedImageCount_;
		if (imageCount < capabilities.minImageCount) {
			imageCount = capabilities.minImageCount;
			std::cerr << "GraphicsPipeline: Requested image count too low, using minimum: " << imageCount << std::endl;
		}
		if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
			imageCount = capabilities.maxImageCount;
			std::cerr << "GraphicsPipeline: Requested image count too high, using maximum: " << imageCount << std::endl;
		}
	} else {
		imageCount = capabilities.minImageCount + 1;
		if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
			imageCount = capabilities.maxImageCount;
		}
	}

	std::cout << "GraphicsPipeline: Creating swapchain with " << imageCount << " images" << std::endl;

	VkSwapchainCreateInfoKHR swapchainInfo{};
	swapchainInfo.sType			   = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainInfo.surface		   = surface_;
	swapchainInfo.minImageCount	   = imageCount;
	swapchainInfo.imageFormat	   = surfaceFormat.format;
	swapchainInfo.imageColorSpace  = surfaceFormat.colorSpace;
	swapchainInfo.imageExtent	   = extent;
	swapchainInfo.imageArrayLayers = 1;
	swapchainInfo.imageUsage	   = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchainInfo.preTransform	   = capabilities.currentTransform;
	swapchainInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainInfo.presentMode	   = presentMode;
	swapchainInfo.clipped		   = VK_TRUE;
	swapchainInfo.oldSwapchain	   = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(gpu_->device, &swapchainInfo, nullptr, &swapchain_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create swapchain" << std::endl;
		return false;
	}

	vkGetSwapchainImagesKHR(gpu_->device, swapchain_, &imageCount, nullptr);
	swapchainImages_.resize(imageCount);
	vkGetSwapchainImagesKHR(gpu_->device, swapchain_, &imageCount, swapchainImages_.data());

	swapchainImageViews_.resize(swapchainImages_.size());
	for (size_t i = 0; i < swapchainImages_.size(); i++) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image							 = swapchainImages_[i];
		viewInfo.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format							 = surfaceFormat.format;
		viewInfo.subresourceRange.aspectMask	 = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel	 = 0;
		viewInfo.subresourceRange.levelCount	 = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount	 = 1;

		if (vkCreateImageView(gpu_->device, &viewInfo, nullptr, &swapchainImageViews_[i]) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create swapchain image view" << std::endl;
			return false;
		}
	}

	width_	= extent.width;
	height_ = extent.height;

	// Initialize image tracking (one fence slot per swapchain image)
	imagesInFlight_.resize(swapchainImages_.size(), VK_NULL_HANDLE);

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (imageAvailableSemaphores_.empty()) {
		// Create acquire semaphores per frame (for vkAcquireNextImageKHR)
		imageAvailableSemaphores_.resize(maxFramesInFlight_);
		inFlightFences_.resize(maxFramesInFlight_);

		for (size_t i = 0; i < maxFramesInFlight_; i++) {
			if (vkCreateSemaphore(gpu_->device, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS) {
				std::cerr << "GraphicsPipeline: Failed to create acquire semaphore for frame " << i << std::endl;
				return false;
			}
		}

		for (size_t i = 0; i < maxFramesInFlight_; i++) {
			if (vkCreateFence(gpu_->device, &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS) {
				std::cerr << "GraphicsPipeline: Failed to create fence for frame " << i << std::endl;
				return false;
			}
		}
	}

	if (renderFinishedSemaphores_.size() != swapchainImages_.size()) {
		for (auto& semaphore : renderFinishedSemaphores_) {
			if (semaphore != VK_NULL_HANDLE) {
				vkDestroySemaphore(gpu_->device, semaphore, nullptr);
			}
		}
		renderFinishedSemaphores_.clear();
		renderFinishedSemaphores_.resize(swapchainImages_.size());

		for (size_t i = 0; i < swapchainImages_.size(); i++) {
			if (vkCreateSemaphore(gpu_->device, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS) {
				std::cerr << "GraphicsPipeline: Failed to create render finished semaphore for image " << i << std::endl;
				return false;
			}
		}

		std::cout << "GraphicsPipeline: Created " << swapchainImages_.size() << " render finished semaphores (per image)" << std::endl;
	}

	return true;
}

bool GraphicsPipeline::recreateSwapchain() {
	if (!gpu_ || !gpu_->device) {
		return false;
	}

	vkDeviceWaitIdle(gpu_->device);
	destroySwapchain();
	if (!createSwapchain()) {
		return false;
	}

	setViewport(width_, height_);

	if (outputTarget_ == OutputTarget::SDL_SURFACE && renderPass_ != VK_NULL_HANDLE) {
		destroyDepthResources();
		if (!createDepthResources()) {
			return false;
		}
		if (!createSwapchainFramebuffers()) {
			return false;
		}
	}

	return true;
}

void GraphicsPipeline::destroySwapchain() {
	if (!gpu_ || !gpu_->device) return;

	for (auto& fb : swapchainFramebuffers_) {
		if (fb != VK_NULL_HANDLE) {
			vkDestroyFramebuffer(gpu_->device, fb, nullptr);
		}
	}
	swapchainFramebuffers_.clear();

	for (auto& iv : swapchainImageViews_) {
		if (iv != VK_NULL_HANDLE) {
			vkDestroyImageView(gpu_->device, iv, nullptr);
		}
	}
	swapchainImageViews_.clear();
	swapchainImages_.clear();

	if (swapchain_ != VK_NULL_HANDLE) {
		vkDestroySwapchainKHR(gpu_->device, swapchain_, nullptr);
		swapchain_ = VK_NULL_HANDLE;
	}
}

bool GraphicsPipeline::createDepthResources() {
	if (!gpu_ || !gpu_->device) {
		return false;
	}

	VkImageCreateInfo depthImageInfo{};
	depthImageInfo.sType		 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	depthImageInfo.imageType	 = VK_IMAGE_TYPE_2D;
	depthImageInfo.extent.width	 = width_;
	depthImageInfo.extent.height = height_;
	depthImageInfo.extent.depth	 = 1;
	depthImageInfo.mipLevels	 = 1;
	depthImageInfo.arrayLayers	 = 1;
	depthImageInfo.format		 = depthFormat_;
	depthImageInfo.tiling		 = VK_IMAGE_TILING_OPTIMAL;
	depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthImageInfo.usage		 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	depthImageInfo.samples		 = multisampling_.rasterizationSamples;
	depthImageInfo.sharingMode	 = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateImage(gpu_->device, &depthImageInfo, nullptr, &depthImage_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create depth image" << std::endl;
		return false;
	}

	VkMemoryRequirements depthMemRequirements;
	vkGetImageMemoryRequirements(gpu_->device, depthImage_, &depthMemRequirements);

	VkMemoryAllocateInfo depthAllocInfo{};
	depthAllocInfo.sType		  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	depthAllocInfo.allocationSize = depthMemRequirements.size;
	depthAllocInfo.memoryTypeIndex =
			device::findMemoryType(gpu_->physicalDevice, depthMemRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	if (vkAllocateMemory(gpu_->device, &depthAllocInfo, nullptr, &depthImageMemory_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to allocate depth image memory" << std::endl;
		return false;
	}

	vkBindImageMemory(gpu_->device, depthImage_, depthImageMemory_, 0);

	VkImageViewCreateInfo depthViewInfo{};
	depthViewInfo.sType				 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthViewInfo.image				 = depthImage_;
	depthViewInfo.viewType			 = VK_IMAGE_VIEW_TYPE_2D;
	depthViewInfo.format			 = depthFormat_;
	depthViewInfo.subresourceRange.aspectMask	 = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthViewInfo.subresourceRange.baseMipLevel	 = 0;
	depthViewInfo.subresourceRange.levelCount	 = 1;
	depthViewInfo.subresourceRange.baseArrayLayer = 0;
	depthViewInfo.subresourceRange.layerCount	 = 1;

	if (vkCreateImageView(gpu_->device, &depthViewInfo, nullptr, &depthImageView_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create depth image view" << std::endl;
		return false;
	}

	return true;
}

void GraphicsPipeline::destroyDepthResources() {
	if (!gpu_ || !gpu_->device) {
		return;
	}

	if (depthImageView_ != VK_NULL_HANDLE) {
		vkDestroyImageView(gpu_->device, depthImageView_, nullptr);
		depthImageView_ = VK_NULL_HANDLE;
	}

	if (depthImage_ != VK_NULL_HANDLE) {
		vkDestroyImage(gpu_->device, depthImage_, nullptr);
		depthImage_ = VK_NULL_HANDLE;
	}

	if (depthImageMemory_ != VK_NULL_HANDLE) {
		vkFreeMemory(gpu_->device, depthImageMemory_, nullptr);
		depthImageMemory_ = VK_NULL_HANDLE;
	}
}

bool GraphicsPipeline::createSwapchainFramebuffers() {
	if (!gpu_ || !gpu_->device) {
		return false;
	}
	if (renderPass_ == VK_NULL_HANDLE) {
		return true;
	}
	if (swapchainImageViews_.empty() || depthImageView_ == VK_NULL_HANDLE) {
		std::cerr << "GraphicsPipeline: Cannot create swapchain framebuffers" << std::endl;
		return false;
	}

	for (auto& fb : swapchainFramebuffers_) {
		if (fb != VK_NULL_HANDLE) {
			vkDestroyFramebuffer(gpu_->device, fb, nullptr);
		}
	}
	swapchainFramebuffers_.clear();

	swapchainFramebuffers_.resize(swapchainImageViews_.size());
	for (size_t i = 0; i < swapchainImageViews_.size(); i++) {
		std::vector<VkImageView> attachments = {swapchainImageViews_[i], depthImageView_};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType			 = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.renderPass		 = renderPass_;
		fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		fbInfo.pAttachments	 = attachments.data();
		fbInfo.width		 = width_;
		fbInfo.height		 = height_;
		fbInfo.layers		 = 1;

		if (vkCreateFramebuffer(gpu_->device, &fbInfo, nullptr, &swapchainFramebuffers_[i]) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create swapchain framebuffer" << std::endl;
			return false;
		}
	}

	return true;
}
