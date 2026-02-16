#ifndef RENDER_WINDOW_HPP
#define RENDER_WINDOW_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

// Forward declarations pour SDL
struct SDL_Window;

namespace renderApi {

	class GPUContext;
	class SwapChain;

	struct WindowConfig {
		uint32_t	width	   = 1280;
		uint32_t	height	   = 720;
		std::string title	   = "Render Window";
		bool		resizable  = true;
		bool		fullscreen = false;
		bool		vsync	   = true;
	};

	class RenderWindow {
	  public:
		RenderWindow();
		~RenderWindow();

		// Non-copyable
		RenderWindow(const RenderWindow&)			 = delete;
		RenderWindow& operator=(const RenderWindow&) = delete;

		// Movable
		RenderWindow(RenderWindow&& other) noexcept;
		RenderWindow& operator=(RenderWindow&& other) noexcept;

		// Création
		bool create(GPUContext& context, const WindowConfig& config = {});
		void destroy();

		// Gestion de la fenêtre
		bool shouldClose() const;
		void pollEvents();
		void waitEvents();

		// SwapChain
		bool acquireNextImage();
		bool present();
		bool resize(uint32_t newWidth, uint32_t newHeight);

		// Render pass
		void beginRenderPass(VkCommandBuffer cmd, const VkClearColorValue& clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}});
		void endRenderPass(VkCommandBuffer cmd);

		// Getters
		SDL_Window*	  getSDLWindow() const { return window_; }
		VkSurfaceKHR  getSurface() const { return surface_; }
		SwapChain*	  getSwapChain() const { return swapChain_; }
		VkFramebuffer getCurrentFramebuffer() const;
		VkRenderPass  getRenderPass() const;
		VkExtent2D	  getExtent() const;
		uint32_t	  getWidth() const;
		uint32_t	  getHeight() const;
		uint32_t	  getCurrentImageIndex() const { return currentImageIndex_; }

		// Synchronisation
		VkSemaphore getImageAvailableSemaphore() const { return imageAvailableSemaphore_; }
		VkSemaphore getRenderFinishedSemaphore() const { return renderFinishedSemaphore_; }
		VkFence		getInFlightFence() const { return inFlightFence_; }

		// Etat
		bool isValid() const { return window_ != nullptr && surface_ != VK_NULL_HANDLE; }
		bool wasResized() const { return framebufferResized_; }
		void resetResizedFlag() { framebufferResized_ = false; }

		// Extensions Vulkan nécessaires
		static std::vector<const char*> getRequiredInstanceExtensions();

	  private:
		GPUContext*	 context_;
		SDL_Window*	 window_;
		VkSurfaceKHR surface_;
		SwapChain*	 swapChain_;

		// Synchronisation
		VkSemaphore imageAvailableSemaphore_;
		VkSemaphore renderFinishedSemaphore_;
		VkFence		inFlightFence_;

		// Etat
		uint32_t currentImageIndex_;
		bool	 framebufferResized_;
		bool	 vsync_;

		// Configuration
		WindowConfig config_;



		bool createWindow();
		bool createSurface();
		bool createSwapChain();
		bool createSyncObjects();
		void destroySyncObjects();
	};

} // namespace renderApi

#endif