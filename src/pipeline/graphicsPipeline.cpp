#include "pipeline/graphicsPipeline.hpp"

#include "device/renderDevice.hpp"
#include "graphicsPipeline.hpp"

#include <SDL2/SDL_vulkan.h>
#include <iostream>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace renderApi::gpuTask {

	GraphicsPipeline::GraphicsPipeline(device::GPU* gpu, const std::string& name) : gpu_(gpu), name_(name) {
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
		  colorFormat_(other.colorFormat_), depthFormat_(other.depthFormat_), colorImage_(other.colorImage_), colorImageView_(other.colorImageView_),
		  colorImageMemory_(other.colorImageMemory_), depthImage_(other.depthImage_), depthImageView_(other.depthImageView_),
		  depthImageMemory_(other.depthImageMemory_), width_(other.width_), height_(other.height_) {
		other.vertexShader_		= VK_NULL_HANDLE;
		other.fragmentShader_	= VK_NULL_HANDLE;
		other.pipeline_			= VK_NULL_HANDLE;
		other.pipelineLayout_	= VK_NULL_HANDLE;
		other.renderPass_		= VK_NULL_HANDLE;
		other.framebuffer_		= VK_NULL_HANDLE;
		other.colorImage_		= VK_NULL_HANDLE;
		other.colorImageView_	= VK_NULL_HANDLE;
		other.colorImageMemory_ = VK_NULL_HANDLE;
		other.depthImage_		= VK_NULL_HANDLE;
		other.depthImageView_	= VK_NULL_HANDLE;
		other.depthImageMemory_ = VK_NULL_HANDLE;
	}

	GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept {
		if (this != &other) {
			destroy();
			gpu_			   = other.gpu_;
			name_			   = std::move(other.name_);
			vertexShader_	   = other.vertexShader_;
			fragmentShader_	   = other.fragmentShader_;
			pipeline_		   = other.pipeline_;
			pipelineLayout_	   = other.pipelineLayout_;
			renderPass_		   = other.renderPass_;
			framebuffer_	   = other.framebuffer_;
			vertexInputInfo_   = other.vertexInputInfo_;
			inputAssemblyInfo_ = other.inputAssemblyInfo_;
			viewportInfo_	   = other.viewportInfo_;
			rasterizer_		   = other.rasterizer_;
			multisampling_	   = other.multisampling_;
			depthStencil_	   = other.depthStencil_;
			colorBlending_	   = other.colorBlending_;
			shaderStages_	   = std::move(other.shaderStages_);
			viewport_		   = other.viewport_;
			scissor_		   = other.scissor_;
			colorFormat_	   = other.colorFormat_;
			depthFormat_	   = other.depthFormat_;
			colorImage_		   = other.colorImage_;
			colorImageView_	   = other.colorImageView_;
			colorImageMemory_  = other.colorImageMemory_;
			depthImage_		   = other.depthImage_;
			depthImageView_	   = other.depthImageView_;
			depthImageMemory_  = other.depthImageMemory_;
			width_			   = other.width_;
			height_			   = other.height_;

			other.vertexShader_		= VK_NULL_HANDLE;
			other.fragmentShader_	= VK_NULL_HANDLE;
			other.pipeline_			= VK_NULL_HANDLE;
			other.pipelineLayout_	= VK_NULL_HANDLE;
			other.renderPass_		= VK_NULL_HANDLE;
			other.framebuffer_		= VK_NULL_HANDLE;
			other.colorImage_		= VK_NULL_HANDLE;
			other.colorImageView_	= VK_NULL_HANDLE;
			other.colorImageMemory_ = VK_NULL_HANDLE;
			other.depthImage_		= VK_NULL_HANDLE;
			other.depthImageView_	= VK_NULL_HANDLE;
			other.depthImageMemory_ = VK_NULL_HANDLE;
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

	VkPipelineColorBlendAttachmentState attachmentState_{};

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
		attachmentState_.blendEnable		 = blendEnable;
		attachmentState_.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		attachmentState_.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		attachmentState_.colorBlendOp		 = VK_BLEND_OP_ADD;
		attachmentState_.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		attachmentState_.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		attachmentState_.alphaBlendOp		 = VK_BLEND_OP_ADD;
		attachmentState_.colorWriteMask		 = colorWriteMask;

		colorBlending_.attachmentCount = 1;
		colorBlending_.pAttachments	   = &attachmentState_;
	}

	void GraphicsPipeline::setColorFormat(VkFormat format) { colorFormat_ = format; }

	void GraphicsPipeline::setDepthFormat(VkFormat format) { depthFormat_ = format; }

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

		setViewport(width, height);

		VkAttachmentDescription colorAttachment{};
		colorAttachment.format		   = colorFormat_;
		colorAttachment.samples		   = multisampling_.rasterizationSamples;
		colorAttachment.loadOp		   = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp		   = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout	   = VK_IMAGE_LAYOUT_GENERAL;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format		   = depthFormat_;
		depthAttachment.samples		   = multisampling_.rasterizationSamples;
		depthAttachment.loadOp		   = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp		   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout	   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		std::vector<VkAttachmentDescription> attachments = {colorAttachment, depthAttachment};

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout	  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout	  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount	= 1;
		subpass.pColorAttachments		= &colorAttachmentRef;
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

		VkImageCreateInfo imageInfo{};
		imageInfo.sType			= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType		= VK_IMAGE_TYPE_2D;
		imageInfo.extent.width	= width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth	= 1;
		imageInfo.mipLevels		= 1;
		imageInfo.arrayLayers	= 1;
		imageInfo.format		= colorFormat_;
		imageInfo.tiling		= VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage			= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.samples		= multisampling_.rasterizationSamples;
		imageInfo.sharingMode	= VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(gpu_->device, &imageInfo, nullptr, &colorImage_) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create color image" << std::endl;
			return false;
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(gpu_->device, colorImage_, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize  = memRequirements.size;
		allocInfo.memoryTypeIndex = device::findMemoryType(gpu_->physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(gpu_->device, &allocInfo, nullptr, &colorImageMemory_) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to allocate color image memory" << std::endl;
			return false;
		}

		vkBindImageMemory(gpu_->device, colorImage_, colorImageMemory_, 0);

		VkImageViewCreateInfo colorViewInfo{};
		colorViewInfo.sType							  = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		colorViewInfo.image							  = colorImage_;
		colorViewInfo.viewType						  = VK_IMAGE_VIEW_TYPE_2D;
		colorViewInfo.format						  = colorFormat_;
		colorViewInfo.subresourceRange.aspectMask	  = VK_IMAGE_ASPECT_COLOR_BIT;
		colorViewInfo.subresourceRange.baseMipLevel	  = 0;
		colorViewInfo.subresourceRange.levelCount	  = 1;
		colorViewInfo.subresourceRange.baseArrayLayer = 0;
		colorViewInfo.subresourceRange.layerCount	  = 1;

		if (vkCreateImageView(gpu_->device, &colorViewInfo, nullptr, &colorImageView_) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create color image view" << std::endl;
			return false;
		}

		imageInfo.format = depthFormat_;
		imageInfo.usage	 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		if (vkCreateImage(gpu_->device, &imageInfo, nullptr, &depthImage_) != VK_SUCCESS) {
			std::cerr << "GraphicsPipeline: Failed to create depth image" << std::endl;
			return false;
		}

		vkGetImageMemoryRequirements(gpu_->device, depthImage_, &memRequirements);

		allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize  = memRequirements.size;
		allocInfo.memoryTypeIndex = device::findMemoryType(gpu_->physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(gpu_->device, &allocInfo, nullptr, &depthImageMemory_) != VK_SUCCESS) {
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

		std::vector<VkImageView> imageViews = {colorImageView_, depthImageView_};

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

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		if (descriptorSetLayout != VK_NULL_HANDLE) {
			pipelineLayoutInfo.setLayoutCount = 1;
			pipelineLayoutInfo.pSetLayouts	  = &descriptorSetLayout;
		} else {
			pipelineLayoutInfo.setLayoutCount = 0;
			pipelineLayoutInfo.pSetLayouts	  = nullptr;
		}

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

		return true;
	}

	void GraphicsPipeline::destroy() {
		if (!gpu_ || !gpu_->device) {
			return;
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

		if (colorImageView_ != VK_NULL_HANDLE) {
			vkDestroyImageView(gpu_->device, colorImageView_, nullptr);
			colorImageView_ = VK_NULL_HANDLE;
		}

		if (colorImage_ != VK_NULL_HANDLE) {
			vkDestroyImage(gpu_->device, colorImage_, nullptr);
			colorImage_ = VK_NULL_HANDLE;
		}

		if (colorImageMemory_ != VK_NULL_HANDLE) {
			vkFreeMemory(gpu_->device, colorImageMemory_, nullptr);
			colorImageMemory_ = VK_NULL_HANDLE;
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

		if (vertexShader_ != VK_NULL_HANDLE) {
			vkDestroyShaderModule(gpu_->device, vertexShader_, nullptr);
			vertexShader_ = VK_NULL_HANDLE;
		}

		if (fragmentShader_ != VK_NULL_HANDLE) {
			vkDestroyShaderModule(gpu_->device, fragmentShader_, nullptr);
			fragmentShader_ = VK_NULL_HANDLE;
		}
	}

} // namespace renderApi::gpuTask
