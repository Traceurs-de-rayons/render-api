#ifndef MEMORY_ALLOCATOR_HPP
#define MEMORY_ALLOCATOR_HPP

#include <cstdint>
#include <vulkan/vulkan_core.h>

// Forward declare VMA types to avoid including the full header here
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

namespace renderApi::device {
	struct GPU;
}

namespace renderApi::memory {

	enum class MemoryUsage {
		GPU_ONLY,		// DEVICE_LOCAL, fastest GPU access
		CPU_ONLY,		// HOST_VISIBLE | HOST_COHERENT, no GPU access
		CPU_TO_GPU,		// Staging buffers: CPU writes, GPU reads
		GPU_TO_CPU,		// Readback buffers: GPU writes, CPU reads
		CPU_COPY,		// Frequent CPU writes, GPU reads
		GPU_LAZY		// Lazily allocated GPU memory
	};

	struct AllocationInfo {
		VmaAllocation allocation;
		void*		  mappedData;
		VkDeviceSize  offset;
		VkDeviceSize  size;
	};

	class MemoryAllocator {
	  public:
		MemoryAllocator();
		~MemoryAllocator();

		MemoryAllocator(const MemoryAllocator&)			   = delete;
		MemoryAllocator& operator=(const MemoryAllocator&) = delete;

		bool init(device::GPU* gpu);
		void cleanup();

		// Buffer allocation
		bool allocateBuffer(VkDeviceSize		 size,
							VkBufferUsageFlags	 usage,
							MemoryUsage			 memoryUsage,
							VkBuffer&			 outBuffer,
							AllocationInfo&		 outAllocation);

		void destroyBuffer(VkBuffer buffer, const AllocationInfo& allocation);

		// Image allocation
		bool allocateImage(const VkImageCreateInfo& imageInfo,
						   MemoryUsage				memoryUsage,
						   VkImage&					outImage,
						   AllocationInfo&			outAllocation);

		void destroyImage(VkImage image, const AllocationInfo& allocation);

		// Memory mapping
		void* mapMemory(const AllocationInfo& allocation);
		void  unmapMemory(const AllocationInfo& allocation);

		// Utility
		bool isValid() const { return allocator_ != nullptr; }

		VmaAllocator getHandle() const { return allocator_; }

		// Stats
		void printStats() const;

	  private:
		device::GPU* gpu_;
		VmaAllocator allocator_;
	};

} // namespace renderApi::memory

#endif