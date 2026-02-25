#include "buffer/buffer.hpp"
#include "device/renderDevice.hpp"
#include "pipeline/graphicsPipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace renderApi::gpuTask;

bool GraphicsPipeline::build(VkDescriptorSetLayout descriptorSetLayout, uint32_t width, uint32_t height) {
	if (!gpu_ || !gpu_->device) {
		std::cerr << "GraphicsPipeline: GPU not initialized" << std::endl;
		return false;
	}

	if (shaderStages_.empty()) {
		std::cerr << "GraphicsPipeline: No shaders set" << std::endl;
		return false;
	}

	width_ = height_ = height;

	if (outputTarget_ == OutputTarget::SDL_SURFACE && surface_ != VK_NULL_HANDLE) {
		if (!createSwapchain()) {
			return false;
		}
	}

	setViewport(width, height);

	std::vector<VkAttachmentDescription> colorAttachments(colorAttachmentCount_);
	std::vector<VkAttachmentReference>	 colorAttachmentRefs(colorAttachmentCount_);

	for (uint32_t i = 0; i < colorAttachmentCount_; ++i) {
		colorAttachments[i].format		   = colorFormats_[i];
		colorAttachments[i].samples		   = multisampling_.rasterizationSamples;
		colorAttachments[i].loadOp		   = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachments[i].storeOp		   = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachments[i].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachments[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachments[i].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachments[i].finalLayout	   = (outputTarget_ == OutputTarget::SDL_SURFACE) ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;

		colorAttachmentRefs[i].attachment = i;
		colorAttachmentRefs[i].layout	  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}

	VkAttachmentDescription depthAttachment{};
	depthAttachment.format		   = depthFormat_;
	depthAttachment.samples		   = multisampling_.rasterizationSamples;
	depthAttachment.loadOp		   = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp		   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout	   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	std::vector<VkAttachmentDescription> attachments;
	attachments.insert(attachments.end(), colorAttachments.begin(), colorAttachments.end());
	attachments.push_back(depthAttachment);

	VkAttachmentReference depthAttachmentRef{};
	depthAttachmentRef.attachment = colorAttachmentCount_;
	depthAttachmentRef.layout	  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount	= colorAttachmentCount_;
	subpass.pColorAttachments		= colorAttachmentRefs.data();
	subpass.pDepthStencilAttachment = &depthAttachmentRef;

	VkSubpassDependency dependency{};
	dependency.srcSubpass	 = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass	 = 0;
	dependency.srcStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependency.dstStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType		   = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments	   = attachments.data();
	renderPassInfo.subpassCount	   = 1;
	renderPassInfo.pSubpasses	   = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies   = &dependency;

	if (vkCreateRenderPass(gpu_->device, &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create render pass" << std::endl;
		return false;
	}

	colorImages_.resize(colorAttachmentCount_);
	colorImageViews_.resize(colorAttachmentCount_);
	colorImageMemories_.resize(colorAttachmentCount_);

	for (uint32_t i = 0; i < colorAttachmentCount_; ++i) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType			= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType		= VK_IMAGE_TYPE_2D;
		imageInfo.extent.width	= width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth	= 1;
		imageInfo.mipLevels		= 1;
		imageInfo.arrayLayers	= 1;
		imageInfo.format		= colorFormats_[i];
		imageInfo.tiling		= VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage			= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.samples		= multisampling_.rasterizationSamples;
		imageInfo.sharingMode	= VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(gpu_->device, &imageInfo, nullptr, &colorImages_[i]) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create color image " << i << std::endl;
			return false;
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(gpu_->device, colorImages_[i], &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize  = memRequirements.size;
		allocInfo.memoryTypeIndex = device::findMemoryType(gpu_->physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(gpu_->device, &allocInfo, nullptr, &colorImageMemories_[i]) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to allocate color image memory " << i << std::endl;
			return false;
		}

		vkBindImageMemory(gpu_->device, colorImages_[i], colorImageMemories_[i], 0);

		VkImageViewCreateInfo colorViewInfo{};
		colorViewInfo.sType							  = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		colorViewInfo.image							  = colorImages_[i];
		colorViewInfo.viewType						  = VK_IMAGE_VIEW_TYPE_2D;
		colorViewInfo.format						  = colorFormats_[i];
		colorViewInfo.subresourceRange.aspectMask	  = VK_IMAGE_ASPECT_COLOR_BIT;
		colorViewInfo.subresourceRange.baseMipLevel	  = 0;
		colorViewInfo.subresourceRange.levelCount	  = 1;
		colorViewInfo.subresourceRange.baseArrayLayer = 0;
		colorViewInfo.subresourceRange.layerCount	  = 1;

		if (vkCreateImageView(gpu_->device, &colorViewInfo, nullptr, &colorImageViews_[i]) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create color image view " << i << std::endl;
			return false;
		}
	}

	VkImageCreateInfo depthImageInfo{};
	depthImageInfo.sType		 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	depthImageInfo.imageType	 = VK_IMAGE_TYPE_2D;
	depthImageInfo.extent.width	 = width;
	depthImageInfo.extent.height = height;
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
	depthViewInfo.sType							  = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthViewInfo.image							  = depthImage_;
	depthViewInfo.viewType						  = VK_IMAGE_VIEW_TYPE_2D;
	depthViewInfo.format						  = depthFormat_;
	depthViewInfo.subresourceRange.aspectMask	  = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthViewInfo.subresourceRange.baseMipLevel	  = 0;
	depthViewInfo.subresourceRange.levelCount	  = 1;
	depthViewInfo.subresourceRange.baseArrayLayer = 0;
	depthViewInfo.subresourceRange.layerCount	  = 1;

	if (vkCreateImageView(gpu_->device, &depthViewInfo, nullptr, &depthImageView_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create depth image view" << std::endl;
		return false;
	}

	std::vector<VkImageView> imageViews;
	imageViews.insert(imageViews.end(), colorImageViews_.begin(), colorImageViews_.end());
	imageViews.push_back(depthImageView_);

	VkFramebufferCreateInfo framebufferInfo{};
	framebufferInfo.sType			= VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebufferInfo.renderPass		= renderPass_;
	framebufferInfo.attachmentCount = static_cast<uint32_t>(imageViews.size());
	framebufferInfo.pAttachments	= imageViews.data();
	framebufferInfo.width			= width;
	framebufferInfo.height			= height;
	framebufferInfo.layers			= 1;

	if (vkCreateFramebuffer(gpu_->device, &framebufferInfo, nullptr, &framebuffer_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create framebuffer" << std::endl;
		return false;
	}

	if (!vertexAttributes_.empty() || !vertexBindings_.empty()) {
		vertexInputInfo_.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes_.size());
		vertexInputInfo_.pVertexAttributeDescriptions	 = vertexAttributes_.data();
		vertexInputInfo_.vertexBindingDescriptionCount	 = static_cast<uint32_t>(vertexBindings_.size());
		vertexInputInfo_.pVertexBindingDescriptions		 = vertexBindings_.data();
	} else {
		vertexInputInfo_.vertexAttributeDescriptionCount = 0;
		vertexInputInfo_.pVertexAttributeDescriptions	 = nullptr;
		vertexInputInfo_.vertexBindingDescriptionCount	 = 0;
		vertexInputInfo_.pVertexBindingDescriptions		 = nullptr;
	}

	std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments(colorAttachmentCount_);
	for (uint32_t i = 0; i < colorAttachmentCount_; ++i) {
		colorBlendAttachments[i] = colorBlendAttachment_;
	}
	colorBlending_.attachmentCount = colorAttachmentCount_;
	colorBlending_.pAttachments	   = colorBlendAttachments.data();

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	if (descriptorSetLayout != VK_NULL_HANDLE) {
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts	  = &descriptorSetLayout;
	} else {
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts	  = nullptr;
	}
	pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges_.size());
	pipelineLayoutInfo.pPushConstantRanges	  = pushConstantRanges_.empty() ? nullptr : pushConstantRanges_.data();

	if (vkCreatePipelineLayout(gpu_->device, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create pipeline layout" << std::endl;
		return false;
	}

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType		= VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages_.size());
	pipelineInfo.pStages	= shaderStages_.data();

	pipelineInfo.pVertexInputState	 = &vertexInputInfo_;
	pipelineInfo.pInputAssemblyState = &inputAssemblyInfo_;
	pipelineInfo.pViewportState		 = &viewportInfo_;
	pipelineInfo.pRasterizationState = &rasterizer_;
	pipelineInfo.pMultisampleState	 = &multisampling_;
	pipelineInfo.pDepthStencilState	 = &depthStencil_;
	pipelineInfo.pColorBlendState	 = &colorBlending_;
	pipelineInfo.layout				 = pipelineLayout_;
	pipelineInfo.renderPass			 = renderPass_;
	pipelineInfo.subpass			 = 0;
	pipelineInfo.basePipelineHandle	 = VK_NULL_HANDLE;

	if (vkCreateGraphicsPipelines(gpu_->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create graphics pipeline" << std::endl;
		vkDestroyPipelineLayout(gpu_->device, pipelineLayout_, nullptr);
		pipelineLayout_ = VK_NULL_HANDLE;
		return false;
	}

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateFence(gpu_->device, &fenceInfo, nullptr, &renderFence_) != VK_SUCCESS) {
		std::cerr << "GraphicsPipeline: Failed to create render fence" << std::endl;
		return false;
	}

	if (outputTarget_ == OutputTarget::SDL_SURFACE && surface_ != VK_NULL_HANDLE && swapchainFramebuffers_.empty()) {
		swapchainFramebuffers_.resize(swapchainImageViews_.size());
		for (size_t i = 0; i < swapchainImageViews_.size(); i++) {
			std::vector<VkImageView> attachments = {swapchainImageViews_[i], depthImageView_};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType		   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.renderPass	   = renderPass_;
			fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbInfo.pAttachments	   = attachments.data();
			fbInfo.width		   = width_;
			fbInfo.height		   = height_;
			fbInfo.layers		   = 1;

			if (vkCreateFramebuffer(gpu_->device, &fbInfo, nullptr, &swapchainFramebuffers_[i]) != VK_SUCCESS) {
				std::cerr << "GraphicsPipeline: Failed to create swapchain framebuffer" << std::endl;
				return false;
			}
		}
	}

	return true;
}

void GraphicsPipeline::destroy() {
	if (!gpu_ || !gpu_->device) {
		return;
	}

	if (renderFence_ != VK_NULL_HANDLE) {
		vkDestroyFence(gpu_->device, renderFence_, nullptr);
		renderFence_ = VK_NULL_HANDLE;
	}

	if (framebuffer_ != VK_NULL_HANDLE) {
		vkDestroyFramebuffer(gpu_->device, framebuffer_, nullptr);
		framebuffer_ = VK_NULL_HANDLE;
	}

	if (renderPass_ != VK_NULL_HANDLE) {
		vkDestroyRenderPass(gpu_->device, renderPass_, nullptr);
		renderPass_ = VK_NULL_HANDLE;
	}

	if (pipeline_ != VK_NULL_HANDLE) {
		vkDestroyPipeline(gpu_->device, pipeline_, nullptr);
		pipeline_ = VK_NULL_HANDLE;
	}

	if (pipelineLayout_ != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(gpu_->device, pipelineLayout_, nullptr);
		pipelineLayout_ = VK_NULL_HANDLE;
	}

	for (auto imageView : colorImageViews_) {
		if (imageView != VK_NULL_HANDLE) {
			vkDestroyImageView(gpu_->device, imageView, nullptr);
		}
	}
	colorImageViews_.clear();

	for (auto image : colorImages_) {
		if (image != VK_NULL_HANDLE) {
			vkDestroyImage(gpu_->device, image, nullptr);
		}
	}
	colorImages_.clear();

	for (auto memory : colorImageMemories_) {
		if (memory != VK_NULL_HANDLE) {
			vkFreeMemory(gpu_->device, memory, nullptr);
		}
	}
	colorImageMemories_.clear();

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

	destroySwapchain();

	for (auto& semaphore : imageAvailableSemaphores_) {
		if (semaphore != VK_NULL_HANDLE) {
			vkDestroySemaphore(gpu_->device, semaphore, nullptr);
		}
	}
	imageAvailableSemaphores_.clear();

	for (auto& semaphore : renderFinishedSemaphores_) {
		if (semaphore != VK_NULL_HANDLE) {
			vkDestroySemaphore(gpu_->device, semaphore, nullptr);
		}
	}
	renderFinishedSemaphores_.clear();

	for (auto& fence : inFlightFences_) {
		if (fence != VK_NULL_HANDLE) {
			vkDestroyFence(gpu_->device, fence, nullptr);
		}
	}
	inFlightFences_.clear();

	if (surface_ != VK_NULL_HANDLE) {
		vkDestroySurfaceKHR(gpu_->instance, surface_, nullptr);
		surface_ = VK_NULL_HANDLE;
	}

	if (vertexShader_ != VK_NULL_HANDLE) {
		vkDestroyShaderModule(gpu_->device, vertexShader_, nullptr);
		vertexShader_ = VK_NULL_HANDLE;
	}

	if (fragmentShader_ != VK_NULL_HANDLE) {
		vkDestroyShaderModule(gpu_->device, fragmentShader_, nullptr);
		fragmentShader_ = VK_NULL_HANDLE;
	}
}
