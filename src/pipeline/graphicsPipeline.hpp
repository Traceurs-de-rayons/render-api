#ifndef GRAPHICS_PIPELINE_HPP
#define GRAPHICS_PIPELINE_HPP

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

struct SDL_Window;

namespace renderApi {
	class Buffer;
	enum class BufferType;
} // namespace renderApi

namespace renderApi::device {
	struct GPU;
}

namespace renderApi::gpuTask {

	class GpuTask;

	enum class OutputTarget { BUFFER, SDL_SURFACE };

	class GraphicsPipeline {
	  public:
		GraphicsPipeline(device::GPU* gpu, const std::string& name);

	  private:
		device::GPU* gpu_ = nullptr;
		std::string	 name_;

		VkShaderModule	 vertexShader_	 = VK_NULL_HANDLE;
		VkShaderModule	 fragmentShader_ = VK_NULL_HANDLE;
		VkPipeline		 pipeline_		 = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
		VkRenderPass	 renderPass_	 = VK_NULL_HANDLE;
		VkFramebuffer	 framebuffer_	 = VK_NULL_HANDLE;
		VkFence			 renderFence_	 = VK_NULL_HANDLE;
		std::mutex		 imageMutex_;

		VkPipelineVertexInputStateCreateInfo   vertexInputInfo_{};
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo_{};
		VkPipelineViewportStateCreateInfo	   viewportInfo_{};
		VkPipelineRasterizationStateCreateInfo rasterizer_{};
		VkPipelineMultisampleStateCreateInfo   multisampling_{};
		VkPipelineDepthStencilStateCreateInfo  depthStencil_{};
		VkPipelineColorBlendStateCreateInfo	   colorBlending_{};
		VkPipelineColorBlendAttachmentState	   colorBlendAttachment_{};

		std::vector<VkPipelineShaderStageCreateInfo> shaderStages_;

		std::vector<VkVertexInputAttributeDescription> vertexAttributes_;
		std::vector<VkVertexInputBindingDescription>   vertexBindings_;

		VkViewport viewport_{};
		VkRect2D   scissor_{};

		VkFormat	   colorFormat_		 = VK_FORMAT_R8G8B8A8_UNORM;
		VkFormat	   depthFormat_		 = VK_FORMAT_D32_SFLOAT;
		VkImage		   colorImage_		 = VK_NULL_HANDLE;
		VkImageView	   colorImageView_	 = VK_NULL_HANDLE;
		VkDeviceMemory colorImageMemory_ = VK_NULL_HANDLE;
		VkImage		   depthImage_		 = VK_NULL_HANDLE;
		VkImageView	   depthImageView_	 = VK_NULL_HANDLE;
		VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
		uint32_t	   width_			 = 0;
		uint32_t	   height_			 = 0;

		bool enabled_ = true;

		OutputTarget			 outputTarget_ = OutputTarget::BUFFER;
		SDL_Window*				 window_	   = nullptr;
		VkSurfaceKHR			 surface_	   = VK_NULL_HANDLE;
		VkSwapchainKHR			 swapchain_	   = VK_NULL_HANDLE;
		std::vector<VkImage>	 swapchainImages_;
		std::vector<VkImageView> swapchainImageViews_;
		uint32_t				 currentFrame_ = 0;

		friend class GpuTask;

	  public:
		~GraphicsPipeline();

		GraphicsPipeline(const GraphicsPipeline&)			 = delete;
		GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
		GraphicsPipeline(GraphicsPipeline&& other) noexcept;
		GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;

		void setVertexShader(const std::vector<uint32_t>& spvCode);
		void setFragmentShader(const std::vector<uint32_t>& spvCode);

		void setVertexInputState(const VkPipelineVertexInputStateCreateInfo& vertexInputInfo);
		void addVertexBinding(uint32_t binding, uint32_t stride, VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_VERTEX);
		void addVertexAttribute(uint32_t location, uint32_t binding, VkFormat format, uint32_t offset);
		void setInputAssemblyState(const VkPipelineInputAssemblyStateCreateInfo& inputAssemblyInfo);
		void setViewport(uint32_t width, uint32_t height, float x = 0.0f, float y = 0.0f);
		void setRasterizer(VkPolygonMode   polygonMode = VK_POLYGON_MODE_FILL,
						   VkCullModeFlags cullMode	   = VK_CULL_MODE_BACK_BIT,
						   VkFrontFace	   frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE);
		void setMultisampling(VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
		void setDepthStencil(bool depthTestEnable = true, bool depthWriteEnable = true, VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS);
		void setColorBlendAttachment(bool				   blendEnable	  = false,
									 VkColorComponentFlags colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
																			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
		void setColorFormat(VkFormat format);
		void setDepthFormat(VkFormat format);

		void setOutputTarget(OutputTarget target);
		void setSDLWindow(SDL_Window* window);

		const std::string& getName() const { return name_; }
		VkPipeline		   getPipeline() const { return pipeline_; }
		VkPipelineLayout   getLayout() const { return pipelineLayout_; }
		VkRenderPass	   getRenderPass() const { return renderPass_; }
		VkFramebuffer	   getFramebuffer() const { return framebuffer_; }
		VkImage			   getColorImage() const { return colorImage_; }
		VkFormat		   getColorFormat() const { return colorFormat_; }
		uint32_t		   getWidth() const { return width_; }
		uint32_t		   getHeight() const { return height_; }
		device::GPU*	   getGPU() const { return gpu_; }

		void setEnabled(bool enabled) { enabled_ = enabled; }
		bool isEnabled() const { return enabled_; }

		VkFence getRenderFence() const { return renderFence_; }

		std::optional<Buffer> getOutputImageToBuffer();

		void destroy();

		bool build(VkDescriptorSetLayout descriptorSetLayout, uint32_t width, uint32_t height);
	};

} // namespace renderApi::gpuTask

#endif
