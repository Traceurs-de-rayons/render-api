#include "device/renderDevice.hpp"
#include "pipeline/graphicsPipeline.hpp"

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

	if (imageAvailableSemaphores_.empty()) {
		imageAvailableSemaphores_.resize(maxFramesInFlight_);
		renderFinishedSemaphores_.resize(maxFramesInFlight_);
		inFlightFences_.resize(maxFramesInFlight_);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < maxFramesInFlight_; i++) {
			if (vkCreateSemaphore(gpu_->device, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
				vkCreateSemaphore(gpu_->device, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS ||
				vkCreateFence(gpu_->device, &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS) {
				std::cerr << "GraphicsPipeline: Failed to create synchronization objects for frame " << i << std::endl;
				return false;
			}
		}
		std::cout << "GraphicsPipeline: Created " << maxFramesInFlight_ << " frames in flight sync objects" << std::endl;
	}

	return true;
}

bool GraphicsPipeline::recreateSwapchain() {
	destroySwapchain();
	return createSwapchain();
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
