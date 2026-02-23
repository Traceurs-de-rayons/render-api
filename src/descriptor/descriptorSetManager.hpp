#ifndef DESCRIPTOR_SET_MANAGER_HPP
#define DESCRIPTOR_SET_MANAGER_HPP

#include <cstdint>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {
	class Buffer;
	class Texture;
	class Image;
	class Sampler;
} // namespace renderApi

namespace renderApi::device {
	struct GPU;
}

namespace renderApi::descriptor {

	enum class DescriptorType {
		UNIFORM_BUFFER,
		STORAGE_BUFFER,
		COMBINED_IMAGE_SAMPLER,
		SAMPLED_IMAGE,
		STORAGE_IMAGE,
		SAMPLER
	};

	struct DescriptorBinding {
		uint32_t		 binding;
		DescriptorType	 type;
		uint32_t		 count;
		VkShaderStageFlags stageFlags;
		
		// Resource pointers (only one should be set based on type)
		Buffer*  buffer	 = nullptr;
		Texture* texture = nullptr;
		Image*	 image	 = nullptr;
		Sampler* sampler = nullptr;
	};

	class DescriptorSet {
	  public:
		DescriptorSet();
		~DescriptorSet();

		DescriptorSet(const DescriptorSet&)			  = delete;
		DescriptorSet& operator=(const DescriptorSet&) = delete;
		DescriptorSet(DescriptorSet&& other) noexcept;
		DescriptorSet& operator=(DescriptorSet&& other) noexcept;

		void addBinding(const DescriptorBinding& binding);
		void addBuffer(uint32_t binding, Buffer* buffer, DescriptorType type, VkShaderStageFlags stages);
		void addTexture(uint32_t binding, Texture* texture, VkShaderStageFlags stages);
		void addImage(uint32_t binding, Image* image, DescriptorType type, VkShaderStageFlags stages);
		void addSampler(uint32_t binding, Sampler* sampler, VkShaderStageFlags stages);

		bool build(device::GPU* gpu, VkDescriptorPool pool);
		void update();
		void destroy();

		VkDescriptorSet		  getHandle() const { return descriptorSet_; }
		VkDescriptorSetLayout getLayout() const { return layout_; }
		bool				  isBuilt() const { return descriptorSet_ != VK_NULL_HANDLE; }

		const std::vector<DescriptorBinding>& getBindings() const { return bindings_; }

	  private:
		device::GPU*					gpu_;
		VkDescriptorSet					descriptorSet_;
		VkDescriptorSetLayout			layout_;
		std::vector<DescriptorBinding>	bindings_;

		VkDescriptorType convertDescriptorType(DescriptorType type) const;
	};

	class DescriptorSetManager {
	  public:
		DescriptorSetManager();
		~DescriptorSetManager();

		DescriptorSetManager(const DescriptorSetManager&)			 = delete;
		DescriptorSetManager& operator=(const DescriptorSetManager&) = delete;

		// Create and manage descriptor sets
		DescriptorSet* createSet(uint32_t setIndex = 0);
		DescriptorSet* getSet(uint32_t setIndex);
		void		   removeSet(uint32_t setIndex);
		void		   clearSets();

		// Build all sets and create pool
		bool build(device::GPU* gpu);
		void destroy();

		// Get layouts for pipeline creation
		std::vector<VkDescriptorSetLayout> getLayouts() const;
		std::vector<VkDescriptorSet>	   getDescriptorSets() const;

		uint32_t getSetCount() const { return static_cast<uint32_t>(sets_.size()); }
		bool	 isBuilt() const { return pool_ != VK_NULL_HANDLE; }

	  private:
		device::GPU*					 gpu_;
		VkDescriptorPool				 pool_;
		std::vector<DescriptorSet>		 sets_;
		std::vector<uint32_t>			 setIndices_;

		bool createPool();
	};

} // namespace renderApi::descriptor

#endif