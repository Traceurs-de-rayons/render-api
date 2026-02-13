#ifndef BUFFER_HPP
#define BUFFER_HPP

#include "../device/renderDevice.hpp"

#include <cstring>
#include <memory>
#include <vector>

namespace renderApi {

	// Buffer types
	enum class BufferType {
		VERTEX,		  // Vertex buffer
		INDEX,		  // Index buffer
		UNIFORM,	  // Uniform buffer (UBO)
		STORAGE,	  // Storage buffer (SSBO)
		STAGING,	  // CPU-accessible staging buffer
		TRANSFER_SRC, // Transfer source
		TRANSFER_DST  // Transfer destination
	};

	// Buffer usage hints
	enum class BufferUsage {
		STATIC,	 // Set once, use many times (GPU only)
		DYNAMIC, // Updated occasionally (CPU->GPU)
		STREAM	 // Updated every frame (CPU->GPU)
	};

	// Forward declaration
	class GPUContext;

	/**
	 * @class Buffer
	 * @brief High-level buffer wrapper
	 * @details RAII-based buffer management with automatic memory handling
	 */
	class Buffer {
	  public:
		Buffer();
		~Buffer();

		// Disable copy, enable move
		Buffer(const Buffer&)			 = delete;
		Buffer& operator=(const Buffer&) = delete;
		Buffer(Buffer&& other) noexcept;
		Buffer& operator=(Buffer&& other) noexcept;

		/**
		 * @brief Create a buffer
		 * @param context GPU context to use
		 * @param size Size in bytes
		 * @param type Buffer type
		 * @param usage Usage pattern
		 * @return true on success
		 */
		bool create(GPUContext& context, size_t size, BufferType type, BufferUsage usage = BufferUsage::STATIC);

		/**
		 * @brief Destroy buffer and free memory
		 */
		void destroy();

		/**
		 * @brief Resize buffer (destroys and recreates, data is lost)
		 * @param newSize New size in bytes
		 * @return true on success
		 */
		bool resize(size_t newSize);

		/**
		 * @brief Upload data to buffer (handles staging automatically)
		 * @param data Pointer to data
		 * @param size Size to upload
		 * @param offset Offset in buffer
		 * @return true on success
		 */
		bool upload(const void* data, size_t size, size_t offset = 0);

		/**
		 * @brief Download data from buffer (handles staging automatically)
		 * @param data Pointer to receive data
		 * @param size Size to download
		 * @param offset Offset in buffer
		 * @return true on success
		 */
		bool download(void* data, size_t size, size_t offset = 0);

		/**
		 * @brief Map buffer for CPU access (only for staging/dynamic buffers)
		 * @return Pointer to mapped memory, or nullptr on failure
		 */
		void* map();

		/**
		 * @brief Unmap buffer
		 */
		void unmap();

		/**
		 * @brief Update buffer with new data (convenience for small updates)
		 * @tparam T Data type
		 * @param data Data to upload
		 * @return true on success
		 */
		template <typename T> bool update(const std::vector<T>& data) { return upload(data.data(), data.size() * sizeof(T)); }

		// Getters
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

	/**
	 * @brief Create a vertex buffer
	 * @param context GPU context
	 * @param vertices Vector of vertex data
	 * @return Created buffer
	 */
	template <typename VertexType> Buffer createVertexBuffer(GPUContext& context, const std::vector<VertexType>& vertices) {
		Buffer buffer;
		if (!buffer.create(context, vertices.size() * sizeof(VertexType), BufferType::VERTEX, BufferUsage::STATIC)) {
			return Buffer();
		}
		buffer.upload(vertices.data(), vertices.size() * sizeof(VertexType));
		return buffer;
	}

	/**
	 * @brief Create an index buffer
	 * @param context GPU context
	 * @param indices Vector of index data
	 * @return Created buffer
	 */
	template <typename IndexType> Buffer createIndexBuffer(GPUContext& context, const std::vector<IndexType>& indices) {
		Buffer buffer;
		if (!buffer.create(context, indices.size() * sizeof(IndexType), BufferType::INDEX, BufferUsage::STATIC)) {
			return Buffer();
		}
		buffer.upload(indices.data(), indices.size() * sizeof(IndexType));
		return buffer;
	}

	/**
	 * @brief Create a uniform buffer
	 * @param context GPU context
	 * @param size Size in bytes
	 * @return Created buffer
	 */
	inline Buffer createUniformBuffer(GPUContext& context, size_t size) {
		Buffer buffer;
		buffer.create(context, size, BufferType::UNIFORM, BufferUsage::DYNAMIC);
		return buffer;
	}

	/**
	 * @brief Create a storage buffer
	 * @param context GPU context
	 * @param size Size in bytes
	 * @param usage Usage pattern
	 * @return Created buffer
	 */
	inline Buffer createStorageBuffer(GPUContext& context, size_t size, BufferUsage usage = BufferUsage::STATIC) {
		Buffer buffer;
		buffer.create(context, size, BufferType::STORAGE, usage);
		return buffer;
	}

	/**
	 * @brief Create a staging buffer
	 * @param context GPU context
	 * @param size Size in bytes
	 * @return Created buffer
	 */
	inline Buffer createStagingBuffer(GPUContext& context, size_t size) {
		Buffer buffer;
		buffer.create(context, size, BufferType::STAGING, BufferUsage::STREAM);
		return buffer;
	}

} // namespace renderApi

#endif