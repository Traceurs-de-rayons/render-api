#include "pipeline/graphicsPipeline.hpp"

#include "buffer/buffer.hpp"
#include "device/renderDevice.hpp"
#include "graphicsPipeline.hpp"

#include <SDL2/SDL_vulkan.h>
#include <algorithm>
#include <iostream>
#include <optional>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace renderApi::gpuTask {

	GraphicsPipeline::GraphicsPipeline(device::GPU* gpu, const std::string& name) : gpu_(gpu), name_(name) {
		// Initialize with single color attachment by default
		colorFormats_.push_back(VK_FORMAT_R8G8B8A8_UNORM);
		colorAttachmentCount_ = 1;

		vertexInputInfo_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		inputAssemblyInfo_.sType				  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyInfo_.topology				  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssemblyInfo_.primitiveRestartEnable = VK_FALSE;

		viewport_.x		   = 0.0f;
		viewport_.y		   = 0.0f;
		viewport_.minDepth = 0.0f;
		viewport_.maxDepth = 1.0f;

		scissor_.offset = {0, 0};

		viewportInfo_.sType			= VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo_.viewportCount = 1;
		viewportInfo_.pViewports	= &viewport_;
		viewportInfo_.scissorCount	= 1;
		viewportInfo_.pScissors		= &scissor_;

		rasterizer_.sType					= VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer_.depthClampEnable		= VK_FALSE;
		rasterizer_.rasterizerDiscardEnable = VK_FALSE;
		rasterizer_.polygonMode				= VK_POLYGON_MODE_FILL;
		rasterizer_.lineWidth				= 1.0f;
		rasterizer_.cullMode				= VK_CULL_MODE_BACK_BIT;
		rasterizer_.frontFace				= VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer_.depthBiasEnable			= VK_FALSE;
		rasterizer_.depthBiasConstantFactor = 0.0f;
		rasterizer_.depthBiasClamp			= 0.0f;
		rasterizer_.depthBiasSlopeFactor	= 0.0f;

		multisampling_.sType				 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling_.sampleShadingEnable	 = VK_FALSE;
		multisampling_.rasterizationSamples	 = VK_SAMPLE_COUNT_1_BIT;
		multisampling_.minSampleShading		 = 1.0f;
		multisampling_.pSampleMask			 = nullptr;
		multisampling_.alphaToCoverageEnable = VK_FALSE;
		multisampling_.alphaToOneEnable		 = VK_FALSE;

		depthStencil_.sType					= VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil_.depthTestEnable		= VK_TRUE;
		depthStencil_.depthWriteEnable		= VK_TRUE;
		depthStencil_.depthCompareOp		= VK_COMPARE_OP_LESS;
		depthStencil_.depthBoundsTestEnable = VK_FALSE;
		depthStencil_.stencilTestEnable		= VK_FALSE;
		depthStencil_.front					= {};
		depthStencil_.back					= {};
		depthStencil_.minDepthBounds		= 0.0f;
		depthStencil_.maxDepthBounds		= 1.0f;

		colorBlending_.sType			 = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending_.logicOpEnable	 = VK_FALSE;
		colorBlending_.logicOp			 = VK_LOGIC_OP_COPY;
		colorBlending_.attachmentCount	 = 1;
		colorBlending_.pAttachments		 = nullptr;
		colorBlending_.blendConstants[0] = 0.0f;
		colorBlending_.blendConstants[1] = 0.0f;
		colorBlending_.blendConstants[2] = 0.0f;
		colorBlending_.blendConstants[3] = 0.0f;
	}

	GraphicsPipeline::~GraphicsPipeline() { destroy(); }

	GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept
		: gpu_(other.gpu_), name_(std::move(other.name_)), vertexShader_(other.vertexShader_), fragmentShader_(other.fragmentShader_),
		  pipeline_(other.pipeline_), pipelineLayout_(other.pipelineLayout_), renderPass_(other.renderPass_), framebuffer_(other.framebuffer_),
		  vertexInputInfo_(other.vertexInputInfo_), inputAssemblyInfo_(other.inputAssemblyInfo_), viewportInfo_(other.viewportInfo_),
		  rasterizer_(other.rasterizer_), multisampling_(other.multisampling_), depthStencil_(other.depthStencil_),
		  colorBlending_(other.colorBlending_), shaderStages_(std::move(other.shaderStages_)), viewport_(other.viewport_), scissor_(other.scissor_),
		  depthFormat_(other.depthFormat_), depthImage_(other.depthImage_), depthImageView_(other.depthImageView_),
		  depthImageMemory_(other.depthImageMemory_), width_(other.width_), height_(other.height_), colorFormats_(std::move(other.colorFormats_)),
		  colorImages_(std::move(other.colorImages_)), colorImageViews_(std::move(other.colorImageViews_)),
		  colorImageMemories_(std::move(other.colorImageMemories_)), colorAttachmentCount_(other.colorAttachmentCount_),
		  outputTarget_(other.outputTarget_), window_(other.window_), surface_(other.surface_), swapchain_(other.swapchain_),
		  swapchainImages_(std::move(other.swapchainImages_)), swapchainImageViews_(std::move(other.swapchainImageViews_)),
		  swapchainFramebuffers_(std::move(other.swapchainFramebuffers_)), currentFrame_(other.currentFrame_),
		  imageAvailableSemaphore_(other.imageAvailableSemaphore_), renderFinishedSemaphore_(other.renderFinishedSemaphore_),
		  renderFence_(other.renderFence_), vertexAttributes_(std::move(other.vertexAttributes_)), vertexBindings_(std::move(other.vertexBindings_)),
		  pushConstantRanges_(std::move(other.pushConstantRanges_)) {
		other.vertexShader_		= VK_NULL_HANDLE;
		other.fragmentShader_	= VK_NULL_HANDLE;
		other.pipeline_			= VK_NULL_HANDLE;
		other.pipelineLayout_	= VK_NULL_HANDLE;
		other.renderPass_		= VK_NULL_HANDLE;
		other.framebuffer_		= VK_NULL_HANDLE;
		other.depthImage_		= VK_NULL_HANDLE;
		other.depthImageView_	= VK_NULL_HANDLE;
		other.depthImageMemory_ = VK_NULL_HANDLE;
	}

	GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept {
		if (this != &other) {
			destroy();
			gpu_					 = other.gpu_;
			name_					 = std::move(other.name_);
			vertexShader_			 = other.vertexShader_;
			fragmentShader_			 = other.fragmentShader_;
			pipeline_				 = other.pipeline_;
			pipelineLayout_			 = other.pipelineLayout_;
			renderPass_				 = other.renderPass_;
			framebuffer_			 = other.framebuffer_;
			vertexInputInfo_		 = other.vertexInputInfo_;
			inputAssemblyInfo_		 = other.inputAssemblyInfo_;
			viewportInfo_			 = other.viewportInfo_;
			rasterizer_				 = other.rasterizer_;
			multisampling_			 = other.multisampling_;
			depthStencil_			 = other.depthStencil_;
			colorBlending_			 = other.colorBlending_;
			shaderStages_			 = std::move(other.shaderStages_);
			viewport_				 = other.viewport_;
			scissor_				 = other.scissor_;
			depthFormat_			 = other.depthFormat_;
			depthImage_				 = other.depthImage_;
			depthImageView_			 = other.depthImageView_;
			depthImageMemory_		 = other.depthImageMemory_;
			width_					 = other.width_;
			height_					 = other.height_;
			colorFormats_			 = std::move(other.colorFormats_);
			colorImages_			 = std::move(other.colorImages_);
			colorImageViews_		 = std::move(other.colorImageViews_);
			colorImageMemories_		 = std::move(other.colorImageMemories_);
			colorAttachmentCount_	 = other.colorAttachmentCount_;
			outputTarget_			 = other.outputTarget_;
			window_					 = other.window_;
			surface_				 = other.surface_;
			swapchain_				 = other.swapchain_;
			swapchainImages_		 = std::move(other.swapchainImages_);
			swapchainImageViews_	 = std::move(other.swapchainImageViews_);
			swapchainFramebuffers_	 = std::move(other.swapchainFramebuffers_);
			currentFrame_			 = other.currentFrame_;
			imageAvailableSemaphore_ = other.imageAvailableSemaphore_;
			renderFinishedSemaphore_ = other.renderFinishedSemaphore_;
			renderFence_			 = other.renderFence_;
			vertexAttributes_		 = std::move(other.vertexAttributes_);
			vertexBindings_			 = std::move(other.vertexBindings_);
			pushConstantRanges_		 = std::move(other.pushConstantRanges_);

			other.vertexShader_			   = VK_NULL_HANDLE;
			other.fragmentShader_		   = VK_NULL_HANDLE;
			other.pipeline_				   = VK_NULL_HANDLE;
			other.pipelineLayout_		   = VK_NULL_HANDLE;
			other.renderPass_			   = VK_NULL_HANDLE;
			other.framebuffer_			   = VK_NULL_HANDLE;
			other.depthImage_			   = VK_NULL_HANDLE;
			other.depthImageView_		   = VK_NULL_HANDLE;
			other.depthImageMemory_		   = VK_NULL_HANDLE;
			other.surface_				   = VK_NULL_HANDLE;
			other.swapchain_			   = VK_NULL_HANDLE;
			other.imageAvailableSemaphore_ = VK_NULL_HANDLE;
			other.renderFinishedSemaphore_ = VK_NULL_HANDLE;
			other.renderFence_			   = VK_NULL_HANDLE;
		}
		return *this;
	}

	void GraphicsPipeline::setVertexShader(const std::vector<uint32_t>& spvCode) {
		if (!gpu_) {
			std::cerr << "GraphicsPipeline: GPU not initialized" << std::endl;
			return;
		}

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = spvCode.size() * sizeof(uint32_t);
		createInfo.pCode	= spvCode.data();

		if (vkCreateShaderModule(gpu_->device, &createInfo, nullptr, &vertexShader_) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create vertex shader module" << std::endl;
			return;
		}

		VkPipelineShaderStageCreateInfo stageInfo{};
		stageInfo.sType	 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stageInfo.stage	 = VK_SHADER_STAGE_VERTEX_BIT;
		stageInfo.module = vertexShader_;
		stageInfo.pName	 = "main";

		shaderStages_.push_back(stageInfo);
	}

	void GraphicsPipeline::setFragmentShader(const std::vector<uint32_t>& spvCode) {
		if (!gpu_) {
			std::cerr << "GraphicsPipeline: GPU not initialized" << std::endl;
			return;
		}

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = spvCode.size() * sizeof(uint32_t);
		createInfo.pCode	= spvCode.data();

		if (vkCreateShaderModule(gpu_->device, &createInfo, nullptr, &fragmentShader_) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create fragment shader module" << std::endl;
			return;
		}

		VkPipelineShaderStageCreateInfo stageInfo{};
		stageInfo.sType	 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stageInfo.stage	 = VK_SHADER_STAGE_FRAGMENT_BIT;
		stageInfo.module = fragmentShader_;
		stageInfo.pName	 = "main";

		shaderStages_.push_back(stageInfo);
	}

	void GraphicsPipeline::setVertexInputState(const VkPipelineVertexInputStateCreateInfo& vertexInputInfo) { vertexInputInfo_ = vertexInputInfo; }

	void GraphicsPipeline::setInputAssemblyState(const VkPipelineInputAssemblyStateCreateInfo& inputAssemblyInfo) {
		inputAssemblyInfo_ = inputAssemblyInfo;
	}

	void GraphicsPipeline::setViewport(uint32_t width, uint32_t height, float x, float y) {
		viewport_.x		   = x;
		viewport_.y		   = y;
		viewport_.width	   = static_cast<float>(width);
		viewport_.height   = static_cast<float>(height);
		viewport_.minDepth = 0.0f;
		viewport_.maxDepth = 1.0f;

		scissor_.extent.width  = width;
		scissor_.extent.height = height;

	}

	void GraphicsPipeline::setRasterizer(VkPolygonMode polygonMode, VkCullModeFlags cullMode, VkFrontFace frontFace) {
		rasterizer_.polygonMode = polygonMode;
		rasterizer_.cullMode	= cullMode;
		rasterizer_.frontFace	= frontFace;
	}

	void GraphicsPipeline::setMultisampling(VkSampleCountFlagBits samples) { multisampling_.rasterizationSamples = samples; }

	void GraphicsPipeline::setDepthStencil(bool depthTestEnable, bool depthWriteEnable, VkCompareOp depthCompareOp) {
		depthStencil_.depthTestEnable  = depthTestEnable;
		depthStencil_.depthWriteEnable = depthWriteEnable;
		depthStencil_.depthCompareOp   = depthCompareOp;
	}

	void GraphicsPipeline::addVertexBinding(uint32_t binding, uint32_t stride, VkVertexInputRate inputRate) {
		VkVertexInputBindingDescription desc{};
		desc.binding   = binding;
		desc.stride	   = stride;
		desc.inputRate = inputRate;
		vertexBindings_.push_back(desc);
	}

	void GraphicsPipeline::addVertexAttribute(uint32_t location, uint32_t binding, VkFormat format, uint32_t offset) {
		VkVertexInputAttributeDescription desc{};
		desc.location = location;
		desc.binding  = binding;
		desc.format	  = format;
		desc.offset	  = offset;
		vertexAttributes_.push_back(desc);
	}

	void GraphicsPipeline::setColorBlendAttachment(bool blendEnable, VkColorComponentFlags colorWriteMask) {
		colorBlendAttachment_.blendEnable		 = blendEnable;
		colorBlendAttachment_.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment_.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment_.colorBlendOp		 = VK_BLEND_OP_ADD;
		colorBlendAttachment_.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment_.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment_.alphaBlendOp		 = VK_BLEND_OP_ADD;
		colorBlendAttachment_.colorWriteMask		 = colorWriteMask;

		colorBlending_.attachmentCount = 1;
		colorBlending_.pAttachments	   = &colorBlendAttachment_;

	}

	void GraphicsPipeline::setColorFormat(VkFormat format) {
		if (!colorFormats_.empty()) {
			colorFormats_[0] = format;
		}
	}

	void GraphicsPipeline::setDepthFormat(VkFormat format) { depthFormat_ = format; }

	void GraphicsPipeline::addPushConstantRange(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size) {
		VkPushConstantRange range{};
		range.stageFlags = stageFlags;
		range.offset	 = offset;
		range.size		 = size;
		pushConstantRanges_.push_back(range);
	}

	void GraphicsPipeline::setColorAttachmentCount(uint32_t count) {
		colorAttachmentCount_ = count;
		colorFormats_.resize(count, VK_FORMAT_R8G8B8A8_UNORM);
	}

	void GraphicsPipeline::setColorAttachmentFormat(uint32_t index, VkFormat format) {
		if (index >= colorFormats_.size()) {
			colorFormats_.resize(index + 1, VK_FORMAT_R8G8B8A8_UNORM);
		}
		colorFormats_[index]  = format;
		colorAttachmentCount_ = std::max(colorAttachmentCount_, index + 1);
	}

	void GraphicsPipeline::setOutputTarget(OutputTarget target) { outputTarget_ = target; }

	void GraphicsPipeline::setSDLWindow(SDL_Window* window) {
		window_ = window;
		if (window_ && gpu_->device) {
			if (!SDL_Vulkan_CreateSurface(window_, gpu_->instance, &surface_)) {
				std::cerr << "GraphicsPipeline: Failed to create Vulkan surface from SDL window: " << SDL_GetError() << std::endl;
				surface_ = VK_NULL_HANDLE;
			}
		}
	}

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
			if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
				presentMode = mode;
				break;
			}
		}

		VkExtent2D extent;
		if (capabilities.currentExtent.width != UINT32_MAX) {
			extent = capabilities.currentExtent;
		} else {
			extent		  = {width_, height_};
			extent.width  = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
		}

		uint32_t imageCount = capabilities.minImageCount + 1;
		if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
			imageCount = capabilities.maxImageCount;
		}

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

		if (renderFinishedSemaphore_ == VK_NULL_HANDLE) {
			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			VkResult result		= vkCreateSemaphore(gpu_->device, &semaphoreInfo, nullptr, &renderFinishedSemaphore_);
			if (result != VK_SUCCESS) {
				std::cerr << "GraphicsPipeline: Failed to create render finished semaphore: " << result << std::endl;
			}
		}
		if (imageAvailableSemaphore_ == VK_NULL_HANDLE) {
			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			VkResult result		= vkCreateSemaphore(gpu_->device, &semaphoreInfo, nullptr, &imageAvailableSemaphore_);
			if (result != VK_SUCCESS) {
				std::cerr << "GraphicsPipeline: Failed to create image available semaphore: " << result << std::endl;
			}
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

	VkFramebuffer GraphicsPipeline::getSwapchainFramebuffer(uint32_t index) const {
		if (index < swapchainFramebuffers_.size()) {
			return swapchainFramebuffers_[index];
		}
		return VK_NULL_HANDLE;
	}

	uint32_t GraphicsPipeline::getSwapchainImageCount() const { return static_cast<uint32_t>(swapchainImages_.size()); }

	VkImage GraphicsPipeline::getSwapchainImage(uint32_t index) const {
		if (index < swapchainImages_.size()) {
			return swapchainImages_[index];
		}
		return VK_NULL_HANDLE;
	}

	bool GraphicsPipeline::build(VkDescriptorSetLayout descriptorSetLayout, uint32_t width, uint32_t height) {
		if (!gpu_ || !gpu_->device) {
			std::cerr << "GraphicsPipeline: GPU not initialized" << std::endl;
			return false;
		}

		if (shaderStages_.empty()) {
			std::cerr << "GraphicsPipeline: No shaders set" << std::endl;
			return false;
		}

		width_	= width;
		height_ = height;

		if (outputTarget_ == OutputTarget::SDL_SURFACE && surface_ != VK_NULL_HANDLE) {
			if (!createSwapchain()) {
				return false;
			}
		}

		setViewport(width, height);

		// Create color attachments for MRT
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
			// Use PRESENT_SRC_KHR for swapchain, GENERAL for offscreen rendering
			colorAttachments[i].finalLayout	   = (outputTarget_ == OutputTarget::SDL_SURFACE)
												   ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
												   : VK_IMAGE_LAYOUT_GENERAL;

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

		// Create color images for MRT
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
			allocInfo.sType			 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex =
					device::findMemoryType(gpu_->physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

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

		// Setup color blend attachments for MRT
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
		pipelineInfo.sType				 = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount			 = static_cast<uint32_t>(shaderStages_.size());
		pipelineInfo.pStages			 = shaderStages_.data();

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


		// Create fence for synchronization
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

		if (renderFinishedSemaphore_ != VK_NULL_HANDLE) {
			vkDestroySemaphore(gpu_->device, renderFinishedSemaphore_, nullptr);
			renderFinishedSemaphore_ = VK_NULL_HANDLE;
		}

		if (imageAvailableSemaphore_ != VK_NULL_HANDLE) {
			vkDestroySemaphore(gpu_->device, imageAvailableSemaphore_, nullptr);
			imageAvailableSemaphore_ = VK_NULL_HANDLE;
		}

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

	std::optional<Buffer> GraphicsPipeline::getOutputImageToBuffer() {
		if (!gpu_ || !gpu_->device) {
			std::cerr << "Cannot get output image: pipeline not initialized" << std::endl;
			return std::nullopt;
		}

		std::lock_guard<std::mutex> lock(imageMutex_);

		VkDeviceSize imageSize = width_ * height_ * 4;
		Buffer		 outputBuffer;
		if (!outputBuffer.create(gpu_, imageSize, BufferType::STAGING, BufferUsage::STREAM, BufferMemory::HOST_VISIBLE)) {
			std::cerr << "Failed to create output buffer" << std::endl;
			return std::nullopt;
		}

		VkCommandBuffer cmdBuffer = gpu_->beginOneTimeCommands();

		VkImage outputImage = !colorImages_.empty() ? colorImages_[0] : VK_NULL_HANDLE;
		if (outputImage == VK_NULL_HANDLE) {
			std::cerr << "GraphicsPipeline: No color image to copy from" << std::endl;
			return std::nullopt;
		}

		VkImageMemoryBarrier barrier{};
		barrier.sType							= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout						= VK_IMAGE_LAYOUT_GENERAL;
		barrier.newLayout						= VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
		barrier.image							= outputImage;
		barrier.subresourceRange.aspectMask		= VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel	= 0;
		barrier.subresourceRange.levelCount		= 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount		= 1;
		barrier.srcAccessMask					= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask					= VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(
				cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		// Copy image to buffer
		VkBufferImageCopy region{};
		region.bufferOffset					   = 0;
		region.bufferRowLength				   = 0;
		region.bufferImageHeight			   = 0;
		region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel	   = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount	   = 1;
		region.imageOffset					   = {0, 0, 0};
		region.imageExtent					   = {width_, height_, 1};

		vkCmdCopyImageToBuffer(cmdBuffer, outputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outputBuffer.getHandle(), 1, &region);

		barrier.oldLayout	  = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout	  = VK_IMAGE_LAYOUT_GENERAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		vkCmdPipelineBarrier(
				cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		gpu_->endOneTimeCommands(cmdBuffer);

		return std::move(outputBuffer);
	}

} // namespace renderApi::gpuTask
