/** @file compute.hpp
 * @brief High-level compute API
 * @details Simple, developer-friendly API for GPU compute operations
 */

#ifndef COMPUTE_HPP
#define COMPUTE_HPP

#include "renderDevice.hpp"

#include <functional>
#include <string>
#include <vector>

namespace renderApi::compute {

	// Forward declarations
	class ComputeContext;
	class Buffer;
	class Pipeline;

	// Buffer types
	enum class BufferType {
		STORAGE, // Read/write buffer (SSBO)
		UNIFORM, // Read-only uniform buffer (UBO)
		STAGING	 // CPU-accessible staging buffer
	};

	// Buffer usage flags
	enum class BufferUsage {
		GPU_ONLY,	 // Device local, fastest for GPU
		CPU_TO_GPU,	 // Host visible, for uploading data
		GPU_TO_CPU,	 // Host visible, for reading back
		CPU_GPU_BOTH // Both directions (slower)
	};

	// Simple Buffer class
	class Buffer {
	  public:
		Buffer() = default;
		~Buffer();

		// Disable copy, enable move
		Buffer(const Buffer&)			 = delete;
		Buffer& operator=(const Buffer&) = delete;
		Buffer(Buffer&& other) noexcept;
		Buffer& operator=(Buffer&& other) noexcept;

		// Create/destroy
		bool create(ComputeContext& context, size_t size, BufferType type, BufferUsage usage);
		void destroy();

		// Resize (destroys and recreates)
		bool resize(size_t newSize);

		// Data transfer
		bool  upload(const void* data, size_t size, size_t offset = 0);
		bool  download(void* data, size_t size, size_t offset = 0);
		void* map();   // Map for CPU access
		void  unmap(); // Unmap

		// Getters
		VkBuffer	   getHandle() const { return buffer_; }
		VkDeviceMemory getMemory() const { return memory_; }
		size_t		   getSize() const { return size_; }
		bool		   isValid() const { return buffer_ != VK_NULL_HANDLE; }
		BufferType	   getType() const { return type_; }

	  private:
		ComputeContext* context_	= nullptr;
		VkBuffer		buffer_		= VK_NULL_HANDLE;
		VkDeviceMemory	memory_		= VK_NULL_HANDLE;
		size_t			size_		= 0;
		BufferType		type_		= BufferType::STORAGE;
		BufferUsage		usage_		= BufferUsage::GPU_ONLY;
		void*			mappedData_ = nullptr;
	};

	// Simple Compute Pipeline class
	class Pipeline {
	  public:
		Pipeline() = default;
		~Pipeline();

		// Disable copy, enable move
		Pipeline(const Pipeline&)			 = delete;
		Pipeline& operator=(const Pipeline&) = delete;
		Pipeline(Pipeline&& other) noexcept;
		Pipeline& operator=(Pipeline&& other) noexcept;

		// Create from SPIR-V bytecode or GLSL source
		bool create(ComputeContext& context, const std::vector<uint32_t>& spirvCode);
		bool createFromGLSL(ComputeContext& context, const std::string& glslCode);

		void destroy();

		// Bind buffers to pipeline
		bool bindBuffer(uint32_t binding, const Buffer& buffer);
		bool bindUniformBuffer(uint32_t binding, const Buffer& buffer);

		// Dispatch compute
		bool dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
		bool dispatchIndirect(const Buffer& indirectBuffer, size_t offset = 0);

		// Getters
		VkPipeline		 getHandle() const { return pipeline_; }
		VkPipelineLayout getLayout() const { return layout_; }
		VkDescriptorSet	 getDescriptorSet() const { return descriptorSet_; }
		bool			 isValid() const { return pipeline_ != VK_NULL_HANDLE; }

	  private:
		bool createDescriptorSetLayout();
		bool createDescriptorSet();
		bool updateDescriptorSet();

		ComputeContext*		  context_			   = nullptr;
		VkShaderModule		  shaderModule_		   = VK_NULL_HANDLE;
		VkPipeline			  pipeline_			   = VK_NULL_HANDLE;
		VkPipelineLayout	  layout_			   = VK_NULL_HANDLE;
		VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
		VkDescriptorSet		  descriptorSet_	   = VK_NULL_HANDLE;

		// Bound resources
		struct BoundBuffer {
			const Buffer*	 buffer;
			VkDescriptorType type;
		};
		std::vector<BoundBuffer> boundBuffers_;
		bool					 descriptorsDirty_ = false;
	};

	// High-level Compute Context
	class ComputeContext {
	  public:
		ComputeContext() = default;
		~ComputeContext();

		// Initialize with existing GPU
		bool initialize(device::GPU& gpu);
		void shutdown();

		// Check if initialized
		bool isInitialized() const { return gpu_ != nullptr; }

		// Get underlying GPU
		device::GPU&  getGPU() { return *gpu_; }
		VkDevice	  getDevice() { return gpu_->device; }
		VkQueue		  getComputeQueue() { return gpu_->computeQueue; }
		VkCommandPool getCommandPool() { return gpu_->commandPool; }

		// Command buffer management
		VkCommandBuffer beginOneTimeCommands();
		void			endOneTimeCommands(VkCommandBuffer cmd);

		// Synchronization
		void waitIdle();

		// Convenience: Create buffer
		Buffer createBuffer(size_t size, BufferType type, BufferUsage usage);
		Buffer createStagingBuffer(size_t size);
		Buffer createStorageBuffer(size_t size);
		Buffer createUniformBuffer(size_t size);

		// Convenience: Create pipeline
		Pipeline createPipeline(const std::vector<uint32_t>& spirvCode);
		Pipeline createPipelineFromGLSL(const std::string& glslCode);

	  private:
		device::GPU* gpu_		   = nullptr;
		VkFence		 computeFence_ = VK_NULL_HANDLE;
		bool		 initialized_  = false;
	};

	// Utility functions
	std::vector<uint32_t> compileGLSLToSPIRV(const std::string& glslCode, const std::string& entryPoint = "main");

} // namespace renderApi::compute

#endif