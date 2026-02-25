#include "pipeline/graphicsPipeline.hpp"

#include "buffer/buffer.hpp"
#include "device/renderDevice.hpp"
#include "graphicsPipeline.hpp"
#include "pipeline/pipelineBuild.cpp"
#include "pipeline/swapchain.cpp"

#include <SDL2/SDL_vulkan.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace renderApi::gpuTask;
using namespace renderApi;

GraphicsPipeline::GraphicsPipeline(device::GPU* gpu, const std::string& name) : gpu_(gpu), name_(name) {
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
	  rasterizer_(other.rasterizer_), multisampling_(other.multisampling_), depthStencil_(other.depthStencil_), colorBlending_(other.colorBlending_),
	  shaderStages_(std::move(other.shaderStages_)), viewport_(other.viewport_), scissor_(other.scissor_), depthFormat_(other.depthFormat_),
	  depthImage_(other.depthImage_), depthImageView_(other.depthImageView_), depthImageMemory_(other.depthImageMemory_), width_(other.width_),
	  height_(other.height_), colorFormats_(std::move(other.colorFormats_)), colorImages_(std::move(other.colorImages_)),
	  colorImageViews_(std::move(other.colorImageViews_)), colorImageMemories_(std::move(other.colorImageMemories_)),
	  colorAttachmentCount_(other.colorAttachmentCount_), outputTarget_(other.outputTarget_), window_(other.window_), surface_(other.surface_),
	  swapchain_(other.swapchain_), swapchainImages_(std::move(other.swapchainImages_)), swapchainImageViews_(std::move(other.swapchainImageViews_)),
	  swapchainFramebuffers_(std::move(other.swapchainFramebuffers_)), imageAvailableSemaphores_(std::move(other.imageAvailableSemaphores_)),
	  renderFinishedSemaphores_(std::move(other.renderFinishedSemaphores_)), inFlightFences_(std::move(other.inFlightFences_)),
	  currentFrame_(other.currentFrame_), maxFramesInFlight_(other.maxFramesInFlight_), renderFence_(other.renderFence_),
	  vertexAttributes_(std::move(other.vertexAttributes_)), vertexBindings_(std::move(other.vertexBindings_)),
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
		gpu_					  = other.gpu_;
		name_					  = std::move(other.name_);
		vertexShader_			  = other.vertexShader_;
		fragmentShader_			  = other.fragmentShader_;
		pipeline_				  = other.pipeline_;
		pipelineLayout_			  = other.pipelineLayout_;
		renderPass_				  = other.renderPass_;
		framebuffer_			  = other.framebuffer_;
		vertexInputInfo_		  = other.vertexInputInfo_;
		inputAssemblyInfo_		  = other.inputAssemblyInfo_;
		viewportInfo_			  = other.viewportInfo_;
		rasterizer_				  = other.rasterizer_;
		multisampling_			  = other.multisampling_;
		depthStencil_			  = other.depthStencil_;
		colorBlending_			  = other.colorBlending_;
		shaderStages_			  = std::move(other.shaderStages_);
		viewport_				  = other.viewport_;
		scissor_				  = other.scissor_;
		depthFormat_			  = other.depthFormat_;
		depthImage_				  = other.depthImage_;
		depthImageView_			  = other.depthImageView_;
		depthImageMemory_		  = other.depthImageMemory_;
		width_					  = other.width_;
		height_					  = other.height_;
		colorFormats_			  = std::move(other.colorFormats_);
		colorImages_			  = std::move(other.colorImages_);
		colorImageViews_		  = std::move(other.colorImageViews_);
		colorImageMemories_		  = std::move(other.colorImageMemories_);
		colorAttachmentCount_	  = other.colorAttachmentCount_;
		outputTarget_			  = other.outputTarget_;
		window_					  = other.window_;
		surface_				  = other.surface_;
		swapchain_				  = other.swapchain_;
		swapchainImages_		  = std::move(other.swapchainImages_);
		swapchainImageViews_	  = std::move(other.swapchainImageViews_);
		swapchainFramebuffers_	  = std::move(other.swapchainFramebuffers_);
		imageAvailableSemaphores_ = std::move(other.imageAvailableSemaphores_);
		renderFinishedSemaphores_ = std::move(other.renderFinishedSemaphores_);
		inFlightFences_			  = std::move(other.inFlightFences_);
		currentFrame_			  = other.currentFrame_;
		maxFramesInFlight_		  = other.maxFramesInFlight_;
		renderFence_			  = other.renderFence_;
		vertexAttributes_		  = std::move(other.vertexAttributes_);
		vertexBindings_			  = std::move(other.vertexBindings_);
		pushConstantRanges_		  = std::move(other.pushConstantRanges_);

		other.vertexShader_		= VK_NULL_HANDLE;
		other.fragmentShader_	= VK_NULL_HANDLE;
		other.pipeline_			= VK_NULL_HANDLE;
		other.pipelineLayout_	= VK_NULL_HANDLE;
		other.renderPass_		= VK_NULL_HANDLE;
		other.framebuffer_		= VK_NULL_HANDLE;
		other.depthImage_		= VK_NULL_HANDLE;
		other.depthImageView_	= VK_NULL_HANDLE;
		other.depthImageMemory_ = VK_NULL_HANDLE;
		other.surface_			= VK_NULL_HANDLE;
		other.swapchain_		= VK_NULL_HANDLE;
		other.renderFence_		= VK_NULL_HANDLE;
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
	colorBlendAttachment_.blendEnable		  = blendEnable;
	colorBlendAttachment_.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment_.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment_.colorBlendOp		  = VK_BLEND_OP_ADD;
	colorBlendAttachment_.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment_.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment_.alphaBlendOp		  = VK_BLEND_OP_ADD;
	colorBlendAttachment_.colorWriteMask	  = colorWriteMask;

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

void GraphicsPipeline::setPresentMode(VkPresentModeKHR mode) { preferredPresentMode_ = mode; }

void GraphicsPipeline::setSwapchainImageCount(uint32_t count) { requestedImageCount_ = count; }

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

	vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

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
