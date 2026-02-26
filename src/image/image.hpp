#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "renderDevice.hpp"

#include <cmath>
#include <cstdint>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	class Buffer;

	enum class ImageType { IMAGE_1D, IMAGE_2D, IMAGE_3D, CUBE };

	enum class ImageUsage {
		TEXTURE,		  // Sampled in shaders
		RENDER_TARGET,	  // Color attachment
		DEPTH_STENCIL,	  // Depth/stencil attachment
		STORAGE,		  // Storage image (compute)
		TRANSFER_SRC,	  // Copy source
		TRANSFER_DST,	  // Copy destination
		COMBINED		  // Multiple usages
	};

	enum class ImageLayout {
		UNDEFINED,
		GENERAL,
		COLOR_ATTACHMENT,
		DEPTH_STENCIL_ATTACHMENT,
		SHADER_READ_ONLY,
		TRANSFER_SRC,
		TRANSFER_DST,
		PRESENT_SRC
	};

	struct ImageCreateInfo {
		uint32_t	width			= 1;
		uint32_t	height			= 1;
		uint32_t	depth			= 1;
		uint32_t	mipLevels		= 1;
		uint32_t	arrayLayers		= 1;
		VkFormat	format			= VK_FORMAT_R8G8B8A8_UNORM;
		ImageType	type			= ImageType::IMAGE_2D;
		ImageUsage	usage			= ImageUsage::TEXTURE;
		VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
		bool		generateMipmaps = false;
	};

	class Image {
	  public:
		Image();
		~Image();

		Image(const Image&)			   = delete;
		Image& operator=(const Image&) = delete;
		Image(Image&& other) noexcept;
		Image& operator=(Image&& other) noexcept;

		bool create(device::GPU* gpu, const ImageCreateInfo& info);
		void destroy();

		bool uploadData(const void* data, size_t size);
		bool uploadDataStaged(const void* data, size_t size);

		void transitionLayout(VkCommandBuffer cmd, ImageLayout newLayout);
		void transitionLayout(ImageLayout newLayout);

		void generateMipmaps();

		void copyToBuffer(Buffer& buffer);
		void copyFromBuffer(const Buffer& buffer);

		VkImage		  getHandle() const { return image_; }
		VkImageView	  getView() const { return imageView_; }
		VkDeviceMemory getMemory() const { return memory_; }
		VkFormat	  getFormat() const { return format_; }
		uint32_t	  getWidth() const { return width_; }
		uint32_t	  getHeight() const { return height_; }
		uint32_t	  getDepth() const { return depth_; }
		uint32_t	  getMipLevels() const { return mipLevels_; }
		uint32_t	  getArrayLayers() const { return arrayLayers_; }
		ImageLayout	  getCurrentLayout() const { return currentLayout_; }
		bool		  isValid() const { return image_ != VK_NULL_HANDLE; }

	  private:
		device::GPU*	gpu_;
		VkImage			image_;
		VkImageView		imageView_;
		VkDeviceMemory	memory_;
		VkFormat		format_;
		uint32_t		width_;
		uint32_t		height_;
		uint32_t		depth_;
		uint32_t		mipLevels_;
		uint32_t		arrayLayers_;
		ImageType		type_;
		ImageUsage		usage_;
		ImageLayout		currentLayout_;
		VkImageAspectFlags aspectMask_;

		VkImageUsageFlags getVkUsageFlags() const;
		VkImageAspectFlags getAspectMask() const;
		VkImageLayout	  convertLayout(ImageLayout layout) const;
	};

	enum class FilterMode { NEAREST, LINEAR };

	enum class AddressMode { REPEAT, MIRRORED_REPEAT, CLAMP_TO_EDGE, CLAMP_TO_BORDER };

	struct SamplerCreateInfo {
		FilterMode	magFilter		= FilterMode::LINEAR;
		FilterMode	minFilter		= FilterMode::LINEAR;
		AddressMode addressModeU	= AddressMode::REPEAT;
		AddressMode addressModeV	= AddressMode::REPEAT;
		AddressMode addressModeW	= AddressMode::REPEAT;
		float		maxAnisotropy	= 1.0f;
		bool		enableAnisotropy = false;
		float		minLod			= 0.0f;
		float		maxLod			= VK_LOD_CLAMP_NONE;
		float		mipLodBias		= 0.0f;
	};

	class Sampler {
	  public:
		Sampler();
		~Sampler();

		Sampler(const Sampler&)			  = delete;
		Sampler& operator=(const Sampler&) = delete;
		Sampler(Sampler&& other) noexcept;
		Sampler& operator=(Sampler&& other) noexcept;

		bool create(device::GPU* gpu, const SamplerCreateInfo& info);
		void destroy();

		VkSampler getHandle() const { return sampler_; }
		bool	  isValid() const { return sampler_ != VK_NULL_HANDLE; }

	  private:
		device::GPU* gpu_;
		VkSampler	 sampler_;
	};

	class Texture {
	  public:
		Texture();
		~Texture();

		Texture(const Texture&)			   = delete;
		Texture& operator=(const Texture&) = delete;
		Texture(Texture&& other) noexcept;
		Texture& operator=(Texture&& other) noexcept;

		bool create(device::GPU* gpu, const ImageCreateInfo& imageInfo, const SamplerCreateInfo& samplerInfo);
		bool createFromFile(device::GPU* gpu, const char* filename, bool generateMipmaps = true);
		void destroy();

		bool uploadData(const void* data, size_t size);

		Image&	  getImage() { return image_; }
		Sampler&  getSampler() { return sampler_; }
		VkImageView getImageView() const { return image_.getView(); }
		VkSampler getSamplerHandle() const { return sampler_.getHandle(); }
		bool	  isValid() const { return image_.isValid() && sampler_.isValid(); }

	  private:
		Image	image_;
		Sampler sampler_;
	};

	// Helper functions
	inline Image createImage2D(device::GPU* gpu, uint32_t width, uint32_t height, VkFormat format, ImageUsage usage) {
		ImageCreateInfo info{};
		info.width	= width;
		info.height = height;
		info.format = format;
		info.usage	= usage;
		info.type	= ImageType::IMAGE_2D;

		Image image;
		image.create(gpu, info);
		return image;
	}

	inline Texture createTexture2D(device::GPU*			gpu,
								   uint32_t				width,
								   uint32_t				height,
								   VkFormat				format,
								   const void*			data,
								   size_t				dataSize,
								   bool					generateMipmaps = false) {
		ImageCreateInfo imageInfo{};
		imageInfo.width				= width;
		imageInfo.height			= height;
		imageInfo.format			= format;
		imageInfo.usage				= ImageUsage::TEXTURE;
		imageInfo.type				= ImageType::IMAGE_2D;
		imageInfo.generateMipmaps	= generateMipmaps;
		imageInfo.mipLevels			= generateMipmaps ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1 : 1;

		SamplerCreateInfo samplerInfo{};
		samplerInfo.enableAnisotropy = true;
		samplerInfo.maxAnisotropy	 = 16.0f;

		Texture texture;
		texture.create(gpu, imageInfo, samplerInfo);
		if (data) {
			texture.uploadData(data, dataSize);
		}
		return texture;
	}

} // namespace renderApi

#endif
