#ifndef DESCRIPTORS_HPP
#define DESCRIPTORS_HPP


#include <cstdint>
#include <vector>
#include <vulkan/vulkan_core.h>
namespace renderApi {

	class GPUContext;

	class DescriptorLayout {
	  public:
		DescriptorLayout();
		~DescriptorLayout();

		DescriptorLayout(const DescriptorLayout&)			 = delete;
		DescriptorLayout& operator=(const DescriptorLayout&) = delete;
		DescriptorLayout(DescriptorLayout&& other) noexcept;
		DescriptorLayout& operator=(DescriptorLayout&& other) noexcept;
		DescriptorLayout& addUniformBuffer(uint32_t binding, VkShaderStageFlags stage);
		DescriptorLayout& addStorageBuffer(uint32_t binding, VkShaderStageFlags stage);
		DescriptorLayout& addCombinedImageSampler(uint32_t binding, VkShaderStageFlags stage);

		bool build(GPUContext& context);
		void destroy();

		VkDescriptorSetLayout							 getHandle() const { return layout_; }
		const std::vector<VkDescriptorSetLayoutBinding>& getBindings() const { return bindings_; }
		bool											 isValid() const { return layout_ != VK_NULL_HANDLE; }

	  private:
		GPUContext*								  context_;
		VkDescriptorSetLayout					  layout_;
		std::vector<VkDescriptorSetLayoutBinding> bindings_;
	};

	class DescriptorSet {
	  public:
		DescriptorSet();
		~DescriptorSet();

		DescriptorSet(const DescriptorSet&)			   = delete;
		DescriptorSet& operator=(const DescriptorSet&) = delete;
		DescriptorSet(DescriptorSet&& other) noexcept;
		DescriptorSet& operator=(DescriptorSet&& other) noexcept;

		bool allocate(GPUContext& context, const DescriptorLayout& layout);
		void updateUniformBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
		void updateStorageBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);

		void free();

		VkDescriptorSet getHandle() const { return set_; }
		bool			isValid() const { return set_ != VK_NULL_HANDLE; }

	  private:
		GPUContext*				context_;
		VkDescriptorSet			set_;
		const DescriptorLayout* layout_;
	};

	class DescriptorPool {
	  public:
		DescriptorPool();
		~DescriptorPool();

		DescriptorPool(const DescriptorPool&)			 = delete;
		DescriptorPool& operator=(const DescriptorPool&) = delete;
		DescriptorPool(DescriptorPool&& other) noexcept;
		DescriptorPool& operator=(DescriptorPool&& other) noexcept;

		bool create(GPUContext& context,
					uint32_t	maxSets,
					uint32_t	uniformBuffers		  = 32,
					uint32_t	storageBuffers		  = 32,
					uint32_t	combinedImageSamplers = 32);


		void destroy();
		VkDescriptorPool getHandle() const { return pool_; }
		bool			 isValid() const { return pool_ != VK_NULL_HANDLE; }

	  private:
		GPUContext*		 context_;
		VkDescriptorPool pool_;
	};

}

#endif
