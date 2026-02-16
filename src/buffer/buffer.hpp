#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <cstring>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	enum class BufferType {
		VERTEX,
		INDEX,
		UNIFORM,
		STORAGE,
		STAGING,
		TRANSFER_SRC,
		TRANSFER_DST
	};

	enum class BufferUsage {
		STATIC,
		DYNAMIC,
		STREAM
	};

	class GPUContext;

	class Buffer {
	  public:
		Buffer();
		~Buffer();

		Buffer(const Buffer&)			 = delete;
		Buffer& operator=(const Buffer&) = delete;
		Buffer(Buffer&& other) noexcept;
		Buffer& operator=(Buffer&& other) noexcept;
		bool create(GPUContext& context, size_t size, BufferType type, BufferUsage usage = BufferUsage::STATIC);
		void destroy();
		bool resize(size_t newSize);
		bool upload(const void* data, size_t size, size_t offset = 0);
		bool download(void* data, size_t size, size_t offset = 0);
		void* map();
		void unmap();

		template <typename T> bool update(const std::vector<T>& data) { return upload(data.data(), data.size() * sizeof(T)); }

		VkBuffer		getHandle() const { return buffer_; }
		VkDeviceMemory	getMemory() const { return memory_; }
		VkDeviceAddress getDeviceAddress() const;
		size_t			getSize() const { return size_; }
		BufferType		getType() const { return type_; }
		bool			isValid() const { return buffer_ != VK_NULL_HANDLE; }
		bool			isMapped() const { return mappedPtr_ != nullptr; }

	  private:
		GPUContext*		context_;
		VkBuffer		buffer_;
		VkDeviceMemory	memory_;
		VkDeviceAddress deviceAddress_;
		size_t			size_;
		BufferType		type_;
		BufferUsage		usage_;
		void*			mappedPtr_;
		bool			persistentlyMapped_;

		VkBufferUsageFlags	  getVkUsageFlags() const;
		VkMemoryPropertyFlags getMemoryFlags() const;
	};

	template <typename VertexType> Buffer createVertexBuffer(GPUContext& context, const std::vector<VertexType>& vertices) {
		Buffer buffer;
		if (!buffer.create(context, vertices.size() * sizeof(VertexType), BufferType::VERTEX, BufferUsage::STATIC)) {
			return Buffer();
		}
		buffer.upload(vertices.data(), vertices.size() * sizeof(VertexType));
		return buffer;
	}

	template <typename IndexType> Buffer createIndexBuffer(GPUContext& context, const std::vector<IndexType>& indices) {
		Buffer buffer;
		if (!buffer.create(context, indices.size() * sizeof(IndexType), BufferType::INDEX, BufferUsage::STATIC)) {
			return Buffer();
		}
		buffer.upload(indices.data(), indices.size() * sizeof(IndexType));
		return buffer;
	}

	inline Buffer createUniformBuffer(GPUContext& context, size_t size) {
		Buffer buffer;
		buffer.create(context, size, BufferType::UNIFORM, BufferUsage::DYNAMIC);
		return buffer;
	}

	inline Buffer createStorageBuffer(GPUContext& context, size_t size, BufferUsage usage = BufferUsage::STATIC) {
		Buffer buffer;
		buffer.create(context, size, BufferType::STORAGE, usage);
		return buffer;
	}

	inline Buffer createStagingBuffer(GPUContext& context, size_t size) {
		Buffer buffer;
		buffer.create(context, size, BufferType::STAGING, BufferUsage::STREAM);
		return buffer;
	}

}

#endif
