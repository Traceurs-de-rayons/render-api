#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	class GPUContext;
	class Buffer;

	struct ShaderStage {
		std::vector<uint32_t> spirvCode;
		VkShaderStageFlagBits stage;
		std::string			  entryPoint = "main";
	};

	struct GraphicsPipelineConfig {
		std::vector<ShaderStage> shaderStages;
		std::vector<VkVertexInputBindingDescription>   vertexBindings;
		std::vector<VkVertexInputAttributeDescription> vertexAttributes;

		VkPrimitiveTopology topology			   = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		bool				primitiveRestartEnable = false;

		bool dynamicViewport = true;
		bool dynamicScissor	 = true;

		VkPolygonMode	polygonMode = VK_POLYGON_MODE_FILL;
		VkCullModeFlags cullMode	= VK_CULL_MODE_BACK_BIT;
		VkFrontFace		frontFace	= VK_FRONT_FACE_COUNTER_CLOCKWISE;
		float			lineWidth	= 1.0f;

		VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

		bool		depthTestEnable	  = true;
		bool		depthWriteEnable  = true;
		VkCompareOp depthCompareOp	  = VK_COMPARE_OP_LESS;
		bool		stencilTestEnable = false;

		bool				  blendEnable		  = false;
		VkBlendFactor		  srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		VkBlendFactor		  dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		VkBlendOp			  colorBlendOp		  = VK_BLEND_OP_ADD;
		VkBlendFactor		  srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		VkBlendFactor		  dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		VkBlendOp			  alphaBlendOp		  = VK_BLEND_OP_ADD;
		VkColorComponentFlags colorWriteMask =
				VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkRenderPass renderPass = VK_NULL_HANDLE;
		uint32_t	 subpass	= 0;

		std::vector<VkDescriptorSetLayout> descriptorLayouts;
		std::vector<VkPushConstantRange> pushConstants;
	};

	struct ComputePipelineConfig {
		ShaderStage						   shaderStage;
		std::vector<VkDescriptorSetLayout> descriptorLayouts;
		std::vector<VkPushConstantRange>   pushConstants;
	};

	class GraphicsPipeline {
	  public:
		GraphicsPipeline();
		~GraphicsPipeline();

		GraphicsPipeline(const GraphicsPipeline&)			 = delete;
		GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
		GraphicsPipeline(GraphicsPipeline&& other) noexcept;
		GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;

		bool create(GPUContext& context, const GraphicsPipelineConfig& config);
		void destroy();
		void bind(VkCommandBuffer cmd);
		void bindVertexBuffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset = 0);
		void bindIndexBuffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset = 0, VkIndexType indexType = VK_INDEX_TYPE_UINT32);
		void bindDescriptorSet(VkCommandBuffer cmd, VkDescriptorSet set, uint32_t setIndex = 0);

		template <typename T> void pushConstants(VkCommandBuffer cmd, VkShaderStageFlags stage, uint32_t offset, const T& data) {
			vkCmdPushConstants(cmd, layout_, stage, offset, sizeof(T), &data);
		}

		void draw(VkCommandBuffer cmd, uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t firstInstance = 0);
		void drawIndexed(VkCommandBuffer cmd, uint32_t indexCount, uint32_t	instanceCount = 1, uint32_t firstIndex = 0,int32_t vertexOffset = 0, int32_t firstInstance = 0);

		VkPipeline		 getHandle() const { return pipeline_; }
		VkPipelineLayout getLayout() const { return layout_; }
		bool			 isValid() const { return pipeline_ != VK_NULL_HANDLE; }

	  private:
		GPUContext*					context_;
		VkPipeline					pipeline_;
		VkPipelineLayout			layout_;
		std::vector<VkShaderModule> shaderModules_;

		bool createShaderModules(const std::vector<ShaderStage>& stages);
		void destroyShaderModules();
	};

	class ComputePipeline {
	  public:
		ComputePipeline();
		~ComputePipeline();

		ComputePipeline(const ComputePipeline&)			   = delete;
		ComputePipeline& operator=(const ComputePipeline&) = delete;
		ComputePipeline(ComputePipeline&& other) noexcept;
		ComputePipeline& operator=(ComputePipeline&& other) noexcept;
		bool create(GPUContext& context, const ComputePipelineConfig& config);
		void destroy();
		void bind(VkCommandBuffer cmd);
		void bindDescriptorSet(VkCommandBuffer cmd, VkDescriptorSet set, uint32_t setIndex = 0);

		template <typename T> void pushConstants(VkCommandBuffer cmd, const T& data) {
			vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(T), &data);
		}

		void dispatch(VkCommandBuffer cmd, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

		VkPipeline		 getHandle() const { return pipeline_; }
		VkPipelineLayout getLayout() const { return layout_; }
		bool			 isValid() const { return pipeline_ != VK_NULL_HANDLE; }

	  private:
		GPUContext*		 context_;
		VkPipeline		 pipeline_;
		VkPipelineLayout layout_;
		VkShaderModule	 shaderModule_;
	};

}

#endif
