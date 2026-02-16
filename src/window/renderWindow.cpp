#include "renderWindow.hpp"

#include "gpuContext.hpp"
#include "renderDevice.hpp"
#include "swapchain.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>

using namespace renderApi;



RenderWindow::RenderWindow()
	: context_(nullptr), window_(nullptr), surface_(VK_NULL_HANDLE), swapChain_(nullptr),
	  imageAvailableSemaphore_(VK_NULL_HANDLE), renderFinishedSemaphore_(VK_NULL_HANDLE), inFlightFence_(VK_NULL_HANDLE),
	  currentImageIndex_(0), framebufferResized_(false), vsync_(true) {}

RenderWindow::~RenderWindow() { destroy(); }

RenderWindow::RenderWindow(RenderWindow&& other) noexcept
	: context_(other.context_), window_(other.window_), surface_(other.surface_), swapChain_(other.swapChain_),
	  imageAvailableSemaphore_(other.imageAvailableSemaphore_), renderFinishedSemaphore_(other.renderFinishedSemaphore_),
	  inFlightFence_(other.inFlightFence_), currentImageIndex_(other.currentImageIndex_),
	  framebufferResized_(other.framebufferResized_), vsync_(other.vsync_), config_(other.config_) {
	other.window_					= nullptr;
	other.surface_					= VK_NULL_HANDLE;
	other.swapChain_				= nullptr;
	other.imageAvailableSemaphore_	= VK_NULL_HANDLE;
	other.renderFinishedSemaphore_	= VK_NULL_HANDLE;
	other.inFlightFence_			= VK_NULL_HANDLE;
}

RenderWindow& RenderWindow::operator=(RenderWindow&& other) noexcept {
	if (this != &other) {
		destroy();

		context_				   = other.context_;
		window_					   = other.window_;
		surface_				   = other.surface_;
		swapChain_				   = other.swapChain_;
		imageAvailableSemaphore_   = other.imageAvailableSemaphore_;
		renderFinishedSemaphore_   = other.renderFinishedSemaphore_;
		inFlightFence_			   = other.inFlightFence_;
		currentImageIndex_		   = other.currentImageIndex_;
		framebufferResized_		   = other.framebufferResized_;
		vsync_					   = other.vsync_;
		config_					   = other.config_;

		other.window_					= nullptr;
		other.surface_					= VK_NULL_HANDLE;
		other.swapChain_				= nullptr;
		other.imageAvailableSemaphore_	= VK_NULL_HANDLE;
		other.renderFinishedSemaphore_	= VK_NULL_HANDLE;
		other.inFlightFence_			= VK_NULL_HANDLE;
	}
	return *this;
}

bool RenderWindow::create(GPUContext& context, const WindowConfig& config) {
	context_ = &context;
	config_	 = config;
	vsync_	 = config.vsync;

	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cerr << "[RenderWindow] Failed to initialize SDL: " << SDL_GetError() << std::endl;
		return false;
	}

	if (!createWindow()) {
		return false;
	}

	if (!createSurface()) {
		destroy();
		return false;
	}

	if (!createSwapChain()) {
		destroy();
		return false;
	}

	if (!createSyncObjects()) {
		destroy();
		return false;
	}

	std::cout << "[RenderWindow] Created window: " << config_.title << " (" << config_.width << "x" << config_.height << ")" << std::endl;
	return true;
}

void RenderWindow::destroy() {
	if (context_) {
		context_->waitIdle();
	}

	destroySyncObjects();

	if (swapChain_) {
		delete swapChain_;
		swapChain_ = nullptr;
	}

	if (surface_ != VK_NULL_HANDLE && context_) {
		vkDestroySurfaceKHR(context_->getGPU().instance, surface_, nullptr);
		surface_ = VK_NULL_HANDLE;
	}

	if (window_) {
		SDL_DestroyWindow(window_);
		window_ = nullptr;
	}

	SDL_Quit();
	context_ = nullptr;
}

bool RenderWindow::shouldClose() const {
	if (!window_) return true;
	
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		if (event.type == SDL_QUIT) {
			return true;
		}
		if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
			return true;
		}
	}
	return false;
}

void RenderWindow::pollEvents() {
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
			framebufferResized_ = true;
		}
	}
}

void RenderWindow::waitEvents() { SDL_WaitEvent(nullptr); }

bool RenderWindow::acquireNextImage() {
	if (!swapChain_)
		return false;
	if (!swapChain_->acquireNextImage(imageAvailableSemaphore_, currentImageIndex_)) {// Handle resize
		int width, height;
		SDL_Vulkan_GetDrawableSize(window_, &width, &height);
		if (width > 0 && height > 0)
			resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
		return false;
	}
	return true;
}

bool RenderWindow::present() {
	if (!swapChain_) {
		return false;
	}

	if (!swapChain_->present(renderFinishedSemaphore_, currentImageIndex_)) {// Handle resize
		int width, height;
		SDL_Vulkan_GetDrawableSize(window_, &width, &height);
		if (width > 0 && height > 0)
			resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
		return false;
	}
	return true;
}

bool RenderWindow::resize(uint32_t newWidth, uint32_t newHeight) {
	if (!swapChain_ || !context_) {
		return false;
	}

	context_->waitIdle();

	config_.width  = newWidth;
	config_.height = newHeight;

	framebufferResized_ = true;

	return swapChain_->resize(newWidth, newHeight);
}

void RenderWindow::beginRenderPass(VkCommandBuffer cmd, const VkClearColorValue& clearColor) {
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType			 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass		 = swapChain_->getRenderPass();
	renderPassInfo.framebuffer		 = swapChain_->getFramebuffer(currentImageIndex_);
	renderPassInfo.renderArea.offset = {0, 0};
	renderPassInfo.renderArea.extent = swapChain_->getExtent();

	VkClearValue clearValue{};
	clearValue.color = clearColor;

	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues	   = &clearValue;

	vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void RenderWindow::endRenderPass(VkCommandBuffer cmd) { vkCmdEndRenderPass(cmd); }

VkFramebuffer RenderWindow::getCurrentFramebuffer() const {
	if (swapChain_) {
		return swapChain_->getFramebuffer(currentImageIndex_);
	}
	return VK_NULL_HANDLE;
}

VkRenderPass RenderWindow::getRenderPass() const {
	if (swapChain_) {
		return swapChain_->getRenderPass();
	}
	return VK_NULL_HANDLE;
}

VkExtent2D RenderWindow::getExtent() const {
	if (swapChain_) {
		return swapChain_->getExtent();
	}
	return {0, 0};
}

uint32_t RenderWindow::getWidth() const {
	if (swapChain_) {
		return swapChain_->getWidth();
	}
	return 0;
}

uint32_t RenderWindow::getHeight() const {
	if (swapChain_) {
		return swapChain_->getHeight();
	}
	return 0;
}

std::vector<const char*> RenderWindow::getRequiredInstanceExtensions() {
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cerr << "[RenderWindow] Failed to initialize SDL: " << SDL_GetError() << std::endl;
		return {};
	}

	SDL_Window* dummyWindow = SDL_CreateWindow("", 0, 0, 1, 1, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
	if (!dummyWindow) {
		std::cerr << "[RenderWindow] Failed to create dummy window: " << SDL_GetError() << std::endl;
		return {};
	}

	unsigned int extensionCount = 0;
	if (!SDL_Vulkan_GetInstanceExtensions(dummyWindow, &extensionCount, nullptr)) {
		std::cerr << "[RenderWindow] Failed to get extension count: " << SDL_GetError() << std::endl;
		SDL_DestroyWindow(dummyWindow);
		return {};
	}

	std::vector<const char*> extensions(extensionCount);
	if (!SDL_Vulkan_GetInstanceExtensions(dummyWindow, &extensionCount, extensions.data())) {
		std::cerr << "[RenderWindow] Failed to get extensions: " << SDL_GetError() << std::endl;
		SDL_DestroyWindow(dummyWindow);
		return {};
	}

	SDL_DestroyWindow(dummyWindow);
	return extensions;
}

bool RenderWindow::createWindow() {
	Uint32 flags = SDL_WINDOW_VULKAN;
	
	if (config_.resizable) {
		flags |= SDL_WINDOW_RESIZABLE;
	}
	
	if (config_.fullscreen) {
		flags |= SDL_WINDOW_FULLSCREEN;
	}

	window_ = SDL_CreateWindow(
		config_.title.c_str(),
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		config_.width,
		config_.height,
		flags
	);
	
	if (!window_) {
		std::cerr << "[RenderWindow] Failed to create SDL window: " << SDL_GetError() << std::endl;
		return false;
	}

	return true;
}

bool RenderWindow::createSurface() {
	if (!window_ || !context_) {
		return false;
	}

	VkInstance instance = context_->getGPU().instance;
	if (!SDL_Vulkan_CreateSurface(window_, instance, &surface_)) {
		std::cerr << "[RenderWindow] Failed to create window surface: " << SDL_GetError() << std::endl;
		return false;
	}

	return true;
}

bool RenderWindow::createSwapChain() {
	if (!context_ || surface_ == VK_NULL_HANDLE) {
		return false;
	}

	swapChain_ = new SwapChain();

	SwapChainConfig swapConfig;
	swapConfig.width		= config_.width;
	swapConfig.height		= config_.height;
	swapConfig.imageCount	= 3;
	swapConfig.presentMode	= vsync_ ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_MAILBOX_KHR;

	if (!swapChain_->create(*context_, surface_, swapConfig)) {
		delete swapChain_;
		swapChain_ = nullptr;
		return false;
	}

	return true;
}

bool RenderWindow::createSyncObjects() {
	if (!context_) {
		return false;
	}

	VkDevice device = context_->getDevice();

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore_) != VK_SUCCESS ||
		vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore_) != VK_SUCCESS ||
		vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence_) != VK_SUCCESS) {
		std::cerr << "[RenderWindow] Failed to create synchronization objects!" << std::endl;
		return false;
	}

	return true;
}

void RenderWindow::destroySyncObjects() {
	if (!context_) {
		return;
	}

	VkDevice device = context_->getDevice();

	if (inFlightFence_ != VK_NULL_HANDLE) {
		vkDestroyFence(device, inFlightFence_, nullptr);
		inFlightFence_ = VK_NULL_HANDLE;
	}

	if (renderFinishedSemaphore_ != VK_NULL_HANDLE) {
		vkDestroySemaphore(device, renderFinishedSemaphore_, nullptr);
		renderFinishedSemaphore_ = VK_NULL_HANDLE;
	}

	if (imageAvailableSemaphore_ != VK_NULL_HANDLE) {
		vkDestroySemaphore(device, imageAvailableSemaphore_, nullptr);
		imageAvailableSemaphore_ = VK_NULL_HANDLE;
	}
}
