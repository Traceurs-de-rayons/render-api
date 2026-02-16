#include "swapchain.hpp"

#include "gpuContext.hpp"
#include "renderDevice.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace renderApi {

	SwapChain::SwapChain()
		: context_(nullptr), surface_(VK_NULL_HANDLE), swapChain_(VK_NULL_HANDLE), renderPass_(VK_NULL_HANDLE), extent_{0, 0},
		  format_(VK_FORMAT_UNDEFINED), graphicsQueueFamily_(0), presentQueueFamily_(0) {}

	SwapChain::~SwapChain() { destroy(); }

	SwapChain::SwapChain(SwapChain&& other) noexcept
		: context_(other.context_), surface_(other.surface_), swapChain_(other.swapChain_), renderPass_(other.renderPass_),
		  images_(std::move(other.images_)), imageViews_(std::move(other.imageViews_)), framebuffers_(std::move(other.framebuffers_)),
		  extent_(other.extent_), format_(other.format_), graphicsQueueFamily_(other.graphicsQueueFamily_),
		  presentQueueFamily_(other.presentQueueFamily_) {
		other.swapChain_  = VK_NULL_HANDLE;
		other.renderPass_ = VK_NULL_HANDLE;
		other.surface_	  = VK_NULL_HANDLE;
	}

	SwapChain& SwapChain::operator=(SwapChain&& other) noexcept {
		if (this != &other) {
			destroy();

			context_			   = other.context_;
			surface_			   = other.surface_;
			swapChain_			   = other.swapChain_;
			renderPass_			   = other.renderPass_;
			images_				   = std::move(other.images_);
			imageViews_			   = std::move(other.imageViews_);
			framebuffers_		   = std::move(other.framebuffers_);
			extent_				   = other.extent_;
			format_				   = other.format_;
			graphicsQueueFamily_   = other.graphicsQueueFamily_;
			presentQueueFamily_	   = other.presentQueueFamily_;

			other.swapChain_  = VK_NULL_HANDLE;
			other.renderPass_ = VK_NULL_HANDLE;
			other.surface_	  = VK_NULL_HANDLE;
		}
		return *this;
	}

	bool SwapChain::create(GPUContext& context, VkSurfaceKHR surface, const SwapChainConfig& config) {
		context_ = &context;
		surface_ = surface;

		VkPhysicalDevice physicalDevice = context_->getGPU().physicalDevice;
		VkDevice		 device			= context_->getDevice();
		
		// Get queue families
		auto& queueFamilies = context_->getGPU().queueFamilies;

		// Query swap chain support
		VkSurfaceCapabilitiesKHR		  capabilities;
		std::vector<VkSurfaceFormatKHR>	  formats;
		std::vector<VkPresentModeKHR>	  presentModes;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface_, &capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface_, &formatCount, nullptr);
		if (formatCount != 0) {
			formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface_, &formatCount, formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface_, &presentModeCount, nullptr);
		if (presentModeCount != 0) {
			presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface_, &presentModeCount, presentModes.data());
		}

		// Choose swap chain settings
		VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);
		VkPresentModeKHR   presentMode	 = choosePresentMode(presentModes);
		VkExtent2D		   extent		 = chooseExtent(capabilities, config.width, config.height);

		uint32_t imageCount = config.imageCount;
		if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
			imageCount = capabilities.maxImageCount;
		}
		if (imageCount < capabilities.minImageCount) {
			imageCount = capabilities.minImageCount;
		}

		// Create swap chain
		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType			= VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface			= surface_;
		createInfo.minImageCount	= imageCount;
		createInfo.imageFormat		= surfaceFormat.format;
		createInfo.imageColorSpace	= surfaceFormat.colorSpace;
		createInfo.imageExtent		= extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage		= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		graphicsQueueFamily_ = static_cast<uint32_t>(queueFamilies.graphicsFamily >= 0 ? queueFamilies.graphicsFamily : 0);
		presentQueueFamily_	 = static_cast<uint32_t>(queueFamilies.graphicsFamily >= 0 ? queueFamilies.graphicsFamily : 0);

		uint32_t queueFamilyIndices[] = {graphicsQueueFamily_, presentQueueFamily_};

		if (graphicsQueueFamily_ != presentQueueFamily_) {
			createInfo.imageSharingMode		 = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices	 = queueFamilyIndices;
		} else {
			createInfo.imageSharingMode		 = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices	 = nullptr;
		}

		createInfo.preTransform	  = capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode	  = presentMode;
		createInfo.clipped		  = VK_TRUE;
		createInfo.oldSwapchain	  = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain_) != VK_SUCCESS) {
			std::cerr << "[SwapChain] Failed to create swap chain!" << std::endl;
			return false;
		}

		extent_ = extent;
		format_ = surfaceFormat.format;

		// Retrieve swap chain images
		uint32_t swapChainImageCount;
		vkGetSwapchainImagesKHR(device, swapChain_, &swapChainImageCount, nullptr);
		images_.resize(swapChainImageCount);
		vkGetSwapchainImagesKHR(device, swapChain_, &swapChainImageCount, images_.data());

		// Create image views
		imageViews_.resize(images_.size());
		for (size_t i = 0; i < images_.size(); i++) {
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType							  = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image							  = images_[i];
			viewInfo.viewType						  = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format							  = format_;
			viewInfo.components.r					  = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.g					  = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.b					  = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.a					  = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.subresourceRange.aspectMask	  = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel	  = 0;
			viewInfo.subresourceRange.levelCount	  = 1;
			viewInfo.subresourceRange.baseArrayLayer  = 0;
			viewInfo.subresourceRange.layerCount	  = 1;

			if (vkCreateImageView(device, &viewInfo, nullptr, &imageViews_[i]) != VK_SUCCESS) {
				std::cerr << "[SwapChain] Failed to create image view " << i << "!" << std::endl;
				destroy();
				return false;
			}
		}

		// Create render pass
		if (!createRenderPass()) {
			destroy();
			return false;
		}

		// Create framebuffers
		if (!createFramebuffers()) {
			destroy();
			return false;
		}

		std::cout << "[SwapChain] Created successfully (" << extent_.width << "x" << extent_.height << ")" << std::endl;
		return true;
	}

	void SwapChain::destroy() {
		if (!context_) return;

		VkDevice device = context_->getDevice();

		// Destroy framebuffers
		for (auto framebuffer : framebuffers_) {
			if (framebuffer != VK_NULL_HANDLE) {
				vkDestroyFramebuffer(device, framebuffer, nullptr);
			}
		}
		framebuffers_.clear();

		// Destroy render pass
		if (renderPass_ != VK_NULL_HANDLE) {
			vkDestroyRenderPass(device, renderPass_, nullptr);
			renderPass_ = VK_NULL_HANDLE;
		}

		// Destroy image views
		for (auto imageView : imageViews_) {
			if (imageView != VK_NULL_HANDLE) {
				vkDestroyImageView(device, imageView, nullptr);
			}
		}
		imageViews_.clear();

		// Destroy swap chain
		if (swapChain_ != VK_NULL_HANDLE) {
			vkDestroySwapchainKHR(device, swapChain_, nullptr);
			swapChain_ = VK_NULL_HANDLE;
		}

		images_.clear();
		context_ = nullptr;
		surface_ = VK_NULL_HANDLE;
	}

	bool SwapChain::resize(uint32_t newWidth, uint32_t newHeight) {
		if (!context_ || surface_ == VK_NULL_HANDLE) {
			return false;
		}

		context_->waitIdle();

		SwapChainConfig config;
		config.width  = newWidth;
		config.height = newHeight;


		destroy();
		return create(*context_, surface_, config);
	}

	bool SwapChain::acquireNextImage(VkSemaphore signalSemaphore, uint32_t& imageIndex) {
		VkResult result = vkAcquireNextImageKHR(context_->getDevice(), swapChain_, UINT64_MAX, signalSemaphore, VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			return false;
		} else if (result != VK_SUCCESS) {
			std::cerr << "[SwapChain] Failed to acquire next image!" << std::endl;
			return false;
		}

		return true;
	}

	bool SwapChain::present(VkSemaphore waitSemaphore, uint32_t imageIndex) {
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType			   = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores	   = &waitSemaphore;
		presentInfo.swapchainCount	   = 1;
		presentInfo.pSwapchains		   = &swapChain_;
		presentInfo.pImageIndices	   = &imageIndex;

		VkQueue presentQueue = context_->getGraphicsQueue();
		VkResult result		 = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			return false;
		} else if (result != VK_SUCCESS) {
			std::cerr << "[SwapChain] Failed to present!" << std::endl;
			return false;
		}

		return true;
	}

	VkFramebuffer SwapChain::getFramebuffer(uint32_t imageIndex) const {
		if (imageIndex < framebuffers_.size()) {
			return framebuffers_[imageIndex];
		}
		return VK_NULL_HANDLE;
	}

	bool SwapChain::createRenderPass() {
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format		   = format_;
		colorAttachment.samples		   = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp		   = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp		   = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout	   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout	  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint	 = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments	 = &colorAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass	 = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass	 = 0;
		dependency.srcStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType		   = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments	   = &colorAttachment;
		renderPassInfo.subpassCount	   = 1;
		renderPassInfo.pSubpasses	   = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies   = &dependency;

		if (vkCreateRenderPass(context_->getDevice(), &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS) {
			std::cerr << "[SwapChain] Failed to create render pass!" << std::endl;
			return false;
		}

		return true;
	}

	bool SwapChain::createFramebuffers() {
		framebuffers_.resize(imageViews_.size());

		for (size_t i = 0; i < imageViews_.size(); i++) {
			VkImageView attachments[] = {imageViews_[i]};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType		   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass	   = renderPass_;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments   = attachments;
			framebufferInfo.width		   = extent_.width;
			framebufferInfo.height		   = extent_.height;
			framebufferInfo.layers		   = 1;

			if (vkCreateFramebuffer(context_->getDevice(), &framebufferInfo, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
				std::cerr << "[SwapChain] Failed to create framebuffer " << i << "!" << std::endl;
				return false;
			}
		}

		return true;
	}

	VkSurfaceFormatKHR SwapChain::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
		for (const auto& format : formats) {
			if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return format;
			}
		}
		return formats[0];
	}

	VkPresentModeKHR SwapChain::choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
		for (const auto& mode : modes) {
			if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return mode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D SwapChain::chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		} else {
			VkExtent2D actualExtent = {width, height};

			actualExtent.width	= std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

} // namespace renderApi