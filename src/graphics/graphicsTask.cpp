#include "graphicsTask.hpp"

#include "gpuContext.hpp"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

using namespace renderApi;

GraphicsTask::GraphicsTask()
	: context_(nullptr), window_(nullptr), pipeline_(VK_NULL_HANDLE), pipelineLayout_(VK_NULL_HANDLE), indexBuffer_(nullptr),
	  indexType_(VK_INDEX_TYPE_UINT32), built_(false), viewportX_(0.0f), viewportY_(0.0f), viewportW_(0.0f), viewportH_(0.0f),
	  scissorX_(0), scissorY_(0), scissorW_(0), scissorH_(0), customViewport_(false), customScissor_(false), enabled_(true) {}

GraphicsTask::~GraphicsTask() { destroy(); }

GraphicsTask::GraphicsTask(GraphicsTask&& other) noexcept
	: context_(other.context_), window_(other.window_), name_(std::move(other.name_)), shaders_(std::move(other.shaders_)),
	  pipeline_(other.pipeline_), pipelineLayout_(other.pipelineLayout_), descriptorLayout_(std::move(other.descriptorLayout_)),
	  descriptorSet_(std::move(other.descriptorSet_)), vertexBindings_(std::move(other.vertexBindings_)), indexBuffer_(other.indexBuffer_),
	  indexType_(other.indexType_), uniformBindings_(std::move(other.uniformBindings_)), built_(other.built_), viewportX_(other.viewportX_),
	  viewportY_(other.viewportY_), viewportW_(other.viewportW_), viewportH_(other.viewportH_), scissorX_(other.scissorX_),
	  scissorY_(other.scissorY_), scissorW_(other.scissorW_), scissorH_(other.scissorH_), customViewport_(other.customViewport_),
	  customScissor_(other.customScissor_), enabled_(other.enabled_.load()) {
	other.pipeline_		  = VK_NULL_HANDLE;
	other.pipelineLayout_ = VK_NULL_HANDLE;
	other.built_		  = false;
}

GraphicsTask& GraphicsTask::operator=(GraphicsTask&& other) noexcept {
	if (this != &other) {
		destroy();

		context_		  = other.context_;
		window_			  = other.window_;
		name_			  = std::move(other.name_);
		shaders_		  = std::move(other.shaders_);
		pipeline_		  = other.pipeline_;
		pipelineLayout_	  = other.pipelineLayout_;
		descriptorLayout_ = std::move(other.descriptorLayout_);
		descriptorSet_	  = std::move(other.descriptorSet_);
		vertexBindings_	  = std::move(other.vertexBindings_);
		indexBuffer_	  = other.indexBuffer_;
		indexType_		  = other.indexType_;
		uniformBindings_  = std::move(other.uniformBindings_);
		built_			  = other.built_;
		viewportX_		  = other.viewportX_;
		viewportY_		  = other.viewportY_;
		viewportW_		  = other.viewportW_;
		viewportH_		  = other.viewportH_;
		scissorX_		  = other.scissorX_;
		scissorY_		  = other.scissorY_;
		scissorW_		  = other.scissorW_;
		scissorH_		  = other.scissorH_;
		customViewport_	  = other.customViewport_;
		customScissor_	  = other.customScissor_;
		enabled_.store(other.enabled_.load());

		other.pipeline_		  = VK_NULL_HANDLE;
		other.pipelineLayout_ = VK_NULL_HANDLE;
		other.built_		  = false;
	}
	return *this;
}

bool GraphicsTask::create(GPUContext& context, RenderWindow& window, const std::string& name) {
	context_ = &context;
	window_	 = &window;
	name_	 = name.empty() ? "GraphicsTask" : name;

	return true;
}

void GraphicsTask::destroy() {
	if (!context_) return;

	if (pipeline_ != VK_NULL_HANDLE) {
		vkDestroyPipeline(context_->getDevice(), pipeline_, nullptr);
		pipeline_ = VK_NULL_HANDLE;
	}

	if (pipelineLayout_ != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context_->getDevice(), pipelineLayout_, nullptr);
		pipelineLayout_ = VK_NULL_HANDLE;
	}

	destroyShaderModules();

	descriptorSet_.free();
	descriptorLayout_.destroy();

	vertexBindings_.clear();
	uniformBindings_.clear();
	indexBuffer_ = nullptr;
	built_		 = false;
	context_	 = nullptr;
}

GraphicsTask& GraphicsTask::addShader(ShaderStage stage, const std::vector<uint32_t>& spirvCode, const std::string& name,
									   const std::string& entryPoint) {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot add shader after build()! Call rebuild() instead." << std::endl;
		return *this;
	}

	if (spirvCode.empty()) {
		std::cerr << "[" << name_ << "] Shader SPIR-V code is empty!" << std::endl;
		return *this;
	}

	shaders_[stage] = ShaderModule(stage, spirvCode, name, entryPoint);
	shadersEnabled_[stage] = true;  // Enabled by default
	std::cout << "[" << name_ << "] Added shader stage: " << static_cast<int>(stage) << std::endl;

	return *this;
}

GraphicsTask& GraphicsTask::removeShader(ShaderStage stage) {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot remove shader after build()! Call rebuild() instead." << std::endl;
		return *this;
	}

	auto it = shaders_.find(stage);
	if (it != shaders_.end()) {
		destroyShaderModule(it->second);
		shaders_.erase(it);
		std::cout << "[" << name_ << "] Removed shader stage: " << static_cast<int>(stage) << std::endl;
	}

	return *this;
}

GraphicsTask& GraphicsTask::updateShader(ShaderStage stage, const std::vector<uint32_t>& spirvCode) {
	auto it = shaders_.find(stage);
	if (it == shaders_.end()) {
		std::cerr << "[" << name_ << "] Shader stage not found!" << std::endl;
		return *this;
	}

	if (spirvCode.empty()) {
		std::cerr << "[" << name_ << "] Shader SPIR-V code is empty!" << std::endl;
		return *this;
	}

	// Destroy old module if it exists
	destroyShaderModule(it->second);

	// Update code
	it->second.spirvCode = spirvCode;
	it->second.module	 = VK_NULL_HANDLE;

	// If already built, need to rebuild
	if (built_) {
		std::cout << "[" << name_ << "] Shader updated, rebuilding pipeline..." << std::endl;
		rebuild();
	}

	return *this;
}

bool GraphicsTask::hasShader(ShaderStage stage) const { return shaders_.find(stage) != shaders_.end(); }

void GraphicsTask::clearShaders() {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot clear shaders after build()!" << std::endl;
		return;
	}

	destroyShaderModules();
	shaders_.clear();
	shadersEnabled_.clear();
}

GraphicsTask& GraphicsTask::enableShader(ShaderStage stage) {
	auto it = shaders_.find(stage);
	if (it == shaders_.end()) {
		std::cerr << "[" << name_ << "] Shader stage " << static_cast<int>(stage) << " not found!" << std::endl;
		return *this;
	}

	shadersEnabled_[stage] = true;
	std::cout << "[" << name_ << "] Enabled shader stage " << static_cast<int>(stage) << std::endl;

	// Rebuild if already built
	if (built_) {
		std::cout << "[" << name_ << "] Rebuilding pipeline..." << std::endl;
		rebuild();
	}

	return *this;
}

GraphicsTask& GraphicsTask::disableShader(ShaderStage stage) {
	auto it = shaders_.find(stage);
	if (it == shaders_.end()) {
		std::cerr << "[" << name_ << "] Shader stage " << static_cast<int>(stage) << " not found!" << std::endl;
		return *this;
	}

	// Cannot disable required shaders
	if (stage == ShaderStage::Vertex || stage == ShaderStage::Fragment) {
		std::cerr << "[" << name_ << "] Cannot disable required shader stage (Vertex/Fragment)!" << std::endl;
		return *this;
	}

	shadersEnabled_[stage] = false;
	std::cout << "[" << name_ << "] Disabled shader stage " << static_cast<int>(stage) << std::endl;

	// Rebuild if already built
	if (built_) {
		std::cout << "[" << name_ << "] Rebuilding pipeline..." << std::endl;
		rebuild();
	}

	return *this;
}

bool GraphicsTask::isShaderEnabled(ShaderStage stage) const {
	auto it = shadersEnabled_.find(stage);
	return (it != shadersEnabled_.end()) ? it->second : true;  // Default enabled
}

GraphicsTask& GraphicsTask::bindVertexBuffer(uint32_t binding, Buffer& buffer, uint32_t stride) {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot bind vertex buffer after build()!" << std::endl;
		return *this;
	}

	for (auto& vb : vertexBindings_) {
		if (vb.binding == binding) {
			vb.buffer = &buffer;
			vb.stride = stride;
			return *this;
		}
	}

	vertexBindings_.push_back({binding, &buffer, stride});
	return *this;
}

GraphicsTask& GraphicsTask::bindIndexBuffer(Buffer& buffer, VkIndexType indexType) {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot bind index buffer after build()!" << std::endl;
		return *this;
	}

	indexBuffer_ = &buffer;
	indexType_	 = indexType;
	return *this;
}

GraphicsTask& GraphicsTask::bindUniformBuffer(uint32_t binding, Buffer& buffer) {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot bind uniform buffer after build()!" << std::endl;
		return *this;
	}

	for (auto& ub : uniformBindings_) {
		if (ub.binding == binding) {
			ub.buffer = &buffer;
			return *this;
		}
	}

	uniformBindings_.push_back({binding, &buffer});
	return *this;
}

bool GraphicsTask::build() {
	if (built_) {
		std::cerr << "[" << name_ << "] Already built!" << std::endl;
		return true;
	}

	if (!context_ || !window_) {
		std::cerr << "[" << name_ << "] Context or window is null!" << std::endl;
		return false;
	}

	// Check minimum required shaders
	if (!hasShader(ShaderStage::Vertex)) {
		std::cerr << "[" << name_ << "] Vertex shader is required!" << std::endl;
		return false;
	}

	if (!hasShader(ShaderStage::Fragment)) {
		std::cerr << "[" << name_ << "] Fragment shader is required!" << std::endl;
		return false;
	}

	std::cout << "[" << name_ << "] Building graphics task with " << shaders_.size() << " shaders..." << std::endl;

	if (!createShaderModules())
		return false;

	for (const auto& ub : uniformBindings_)
		descriptorLayout_.addUniformBuffer(ub.binding, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

	if (!uniformBindings_.empty()) {
		if (!descriptorLayout_.build(*context_)) {
			std::cerr << "[" << name_ << "] Failed to build descriptor layout!" << std::endl;
			return false;
		}
	}

	if (!createPipeline(window_->getRenderPass()))
		return false;

	if (!uniformBindings_.empty()) {
		if (!createDescriptors())
			return false;
	}

	built_ = true;
	std::cout << "[" << name_ << "] Build successful!" << std::endl;
	return true;
}

bool GraphicsTask::rebuild() {
	if (!context_ || !window_) {
		std::cerr << "[" << name_ << "] Context or window is null!" << std::endl;
		return false;
	}

	std::cout << "[" << name_ << "] Rebuilding graphics task..." << std::endl;

	// Destroy old pipeline
	if (pipeline_ != VK_NULL_HANDLE) {
		vkDestroyPipeline(context_->getDevice(), pipeline_, nullptr);
		pipeline_ = VK_NULL_HANDLE;
	}

	if (pipelineLayout_ != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context_->getDevice(), pipelineLayout_, nullptr);
		pipelineLayout_ = VK_NULL_HANDLE;
	}

	// Destroy and recreate shader modules
	destroyShaderModules();

	built_ = false;

	return build();
}

void GraphicsTask::setViewport(float x, float y, float width, float height) {
	viewportX_		= x;
	viewportY_		= y;
	viewportW_		= width;
	viewportH_		= height;
	customViewport_ = true;
}

void GraphicsTask::setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
	scissorX_	   = x;
	scissorY_	   = y;
	scissorW_	   = width;
	scissorH_	   = height;
	customScissor_ = true;
}

void GraphicsTask::bind(VkCommandBuffer cmd, VkFramebuffer framebuffer, VkRenderPass renderPass, VkExtent2D extent) {
	if (!isEnabled() || !built_ || pipeline_ == VK_NULL_HANDLE) {
		return;
	}

	// Bind pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

	// Bind descriptor sets if needed
	if (!uniformBindings_.empty()) {
		VkDescriptorSet setHandle = descriptorSet_.getHandle();
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &setHandle, 0, nullptr);
	}

	// Bind vertex buffers
	if (!vertexBindings_.empty()) {
		std::vector<VkBuffer>	  buffers;
		std::vector<VkDeviceSize> offsets;

		for (const auto& vb : vertexBindings_) {
			if (vb.buffer && vb.buffer->isValid()) {
				buffers.push_back(vb.buffer->getHandle());
				offsets.push_back(0);
			}
		}

		if (!buffers.empty()) {
			vkCmdBindVertexBuffers(cmd, 0, static_cast<uint32_t>(buffers.size()), buffers.data(), offsets.data());
		}
	}

	// Bind index buffer if needed
	if (indexBuffer_ && indexBuffer_->isValid()) {
		vkCmdBindIndexBuffer(cmd, indexBuffer_->getHandle(), 0, indexType_);
	}

	// Set viewport
	VkViewport viewport{};
	if (customViewport_) {
		viewport.x		  = viewportX_;
		viewport.y		  = viewportY_;
		viewport.width	  = viewportW_;
		viewport.height	  = viewportH_;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
	} else {
		viewport.x		  = 0.0f;
		viewport.y		  = 0.0f;
		viewport.width	  = static_cast<float>(extent.width);
		viewport.height	  = static_cast<float>(extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
	}
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	// Set scissor
	VkRect2D scissor{};
	if (customScissor_) {
		scissor.offset = {scissorX_, scissorY_};
		scissor.extent = {scissorW_, scissorH_};
	} else {
		scissor.offset = {0, 0};
		scissor.extent = extent;
	}
	vkCmdSetScissor(cmd, 0, 1, &scissor);
}

bool GraphicsTask::createShaderModule(ShaderModule& shader) {
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = shader.spirvCode.size() * sizeof(uint32_t);
	createInfo.pCode	= shader.spirvCode.data();

	if (vkCreateShaderModule(context_->getDevice(), &createInfo, nullptr, &shader.module) != VK_SUCCESS) {
		std::cerr << "[" << name_ << "] Failed to create shader module for stage " << static_cast<int>(shader.stage) << "!" << std::endl;
		return false;
	}

	return true;
}

void GraphicsTask::destroyShaderModule(ShaderModule& shader) {
	if (shader.module != VK_NULL_HANDLE && context_) {
		vkDestroyShaderModule(context_->getDevice(), shader.module, nullptr);
		shader.module = VK_NULL_HANDLE;
	}
}

bool GraphicsTask::createShaderModules() {
	for (auto& [stage, shader] : shaders_) {
		if (!createShaderModule(shader)) {
			return false;
		}
	}
	return true;
}

void GraphicsTask::destroyShaderModules() {
	for (auto& [stage, shader] : shaders_) {
		destroyShaderModule(shader);
	}
}

VkShaderStageFlagBits getVulkanStage(ShaderStage stage) {
	switch (stage) {
		case ShaderStage::Vertex:
			return VK_SHADER_STAGE_VERTEX_BIT;
		case ShaderStage::Fragment:
			return VK_SHADER_STAGE_FRAGMENT_BIT;
		case ShaderStage::Geometry:
			return VK_SHADER_STAGE_GEOMETRY_BIT;
		case ShaderStage::TessellationControl:
			return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
		case ShaderStage::TessellationEvaluation:
			return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		default:
			return VK_SHADER_STAGE_VERTEX_BIT;
	}
}

bool GraphicsTask::createPipeline(VkRenderPass renderPass) {
	// Pipeline layout
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

	if (!uniformBindings_.empty()) {
		layoutInfo.setLayoutCount	   = 1;
		VkDescriptorSetLayout layoutHandle = descriptorLayout_.getHandle();
		layoutInfo.pSetLayouts		   = &layoutHandle;
	} else {
		layoutInfo.setLayoutCount = 0;
		layoutInfo.pSetLayouts	  = nullptr;
	}

	if (vkCreatePipelineLayout(context_->getDevice(), &layoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
		std::cerr << "[" << name_ << "] Failed to create pipeline layout!" << std::endl;
		return false;
	}

	// Shader stages (only include enabled shaders)
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	for (const auto& [stage, shader] : shaders_) {
		// Skip disabled shaders
		if (!isShaderEnabled(stage)) {
			continue;
		}
		
		VkPipelineShaderStageCreateInfo stageInfo{};
		stageInfo.sType	 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stageInfo.stage	 = getVulkanStage(stage);
		stageInfo.module = shader.module;
		stageInfo.pName	 = shader.entryPoint.c_str();
		shaderStages.push_back(stageInfo);
	}

	// Vertex input
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType						   = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount   = 0;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;

	// Input assembly
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType					 = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology				 = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	// Viewport state
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType		   = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount  = 1;

	// Rasterizer
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType				   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable		   = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode			   = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth			   = 1.0f;
	rasterizer.cullMode				   = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace			   = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable		   = VK_FALSE;

	// Multisampling
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType				   = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable  = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	// Color blending
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType			   = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable	   = VK_FALSE;
	colorBlending.logicOp		   = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount  = 1;
	colorBlending.pAttachments	   = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	// Dynamic states
	std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType			   = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates	   = dynamicStates.data();

	// Graphics pipeline
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType				= VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount			= static_cast<uint32_t>(shaderStages.size());
	pipelineInfo.pStages			= shaderStages.data();
	pipelineInfo.pVertexInputState	= &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState		= &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState	= &multisampling;
	pipelineInfo.pColorBlendState	= &colorBlending;
	pipelineInfo.pDynamicState		= &dynamicState;
	pipelineInfo.layout				= pipelineLayout_;
	pipelineInfo.renderPass			= renderPass;
	pipelineInfo.subpass			= 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	if (vkCreateGraphicsPipelines(context_->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS) {
		std::cerr << "[" << name_ << "] Failed to create graphics pipeline!" << std::endl;
		return false;
	}

	return true;
}

bool GraphicsTask::createDescriptors() {
	if (!descriptorSet_.allocate(*context_, descriptorLayout_)) {
		std::cerr << "[" << name_ << "] Failed to allocate descriptor set!" << std::endl;
		return false;
	}

	for (const auto& ub : uniformBindings_) {
		if (ub.buffer && ub.buffer->isValid()) {
			descriptorSet_.updateUniformBuffer(ub.binding, ub.buffer->getHandle());
		}
	}

	return true;
}
