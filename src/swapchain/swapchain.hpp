#ifndef SWAPCHAIN_HPP
#define SWAPCHAIN_HPP

#include <cstdint>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	class GPUContext;
	class RenderWindow;

	struct SwapChainConfig {
		uint32_t		 width			 = 1280;
		uint32_t		 height			 = 720;
		uint32_t		 imageCount		 = 3;
		VkPresentModeKHR presentMode	 = VK_PRESENT_MODE_FIFO_KHR;
		VkFormat		 preferredFormat = VK_FORMAT_B8G8R8A8_UNORM;
		VkColorSpaceKHR	 colorSpace		 = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	};

	class SwapChain {
	  public:
		SwapChain();
		~SwapChain();

		SwapChain(const SwapChain&)			   = delete;
		SwapChain& operator=(const SwapChain&) = delete;
		SwapChain(SwapChain&& other) noexcept;
		SwapChain& operator=(SwapChain&& other) noexcept;
		bool create(GPUContext& context, VkSurfaceKHR surface, const SwapChainConfig& config = {});
		void destroy();
		bool resize(uint32_t newWidth, uint32_t newHeight);
		bool acquireNextImage(VkSemaphore signalSemaphore, uint32_t& imageIndex);
		bool present(VkSemaphore waitSemaphore, uint32_t imageIndex);
		VkFramebuffer getFramebuffer(uint32_t imageIndex) const;
		VkRenderPass getRenderPass() const { return renderPass_; }

		VkSwapchainKHR getHandle() const { return swapChain_; }
		VkExtent2D	   getExtent() const { return extent_; }
		VkFormat	   getFormat() const { return format_; }
		uint32_t	   getImageCount() const { return static_cast<uint32_t>(images_.size()); }
		uint32_t	   getWidth() const { return extent_.width; }
		uint32_t	   getHeight() const { return extent_.height; }
		bool		   isValid() const { return swapChain_ != VK_NULL_HANDLE; }

	  private:
		GPUContext*				   context_;
		VkSurfaceKHR			   surface_;
		VkSwapchainKHR			   swapChain_;
		VkRenderPass			   renderPass_;
		std::vector<VkImage>	   images_;
		std::vector<VkImageView>   imageViews_;
		std::vector<VkFramebuffer> framebuffers_;
		VkExtent2D				   extent_;
		VkFormat				   format_;
		uint32_t				   graphicsQueueFamily_;
		uint32_t				   presentQueueFamily_;

		bool			   createRenderPass();
		bool			   createFramebuffers();
		VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
		VkPresentModeKHR   choosePresentMode(const std::vector<VkPresentModeKHR>& modes);
		VkExtent2D		   chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height);
	};

}

#endif
