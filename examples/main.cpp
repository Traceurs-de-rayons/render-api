#include "buffer/buffer.hpp"
#include "gpuTask/gpuTask.hpp"
#include "image/image.hpp"
#include "objLoader.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "renderApi.hpp"
#include "renderDevice.hpp"
#include "renderInstance.hpp"
#include "utils/utils.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <SDL_error.h>
#include <SDL_events.h>
#include <SDL_keycode.h>
#include <SDL_video.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#ifndef VK_SHADER_STAGE_TASK_BIT_EXT
#define VK_SHADER_STAGE_TASK_BIT_EXT ((VkShaderStageFlagBits)0x00000040)
#endif

#ifndef VK_SHADER_STAGE_MESH_BIT_EXT
#define VK_SHADER_STAGE_MESH_BIT_EXT ((VkShaderStageFlagBits)0x00000080)
#endif

struct Vertex {
	float pos[3];
	float color[3];
	float texCoord[2];
};

struct ModelViewProj {
	float model[16];
	float view[16];
	float proj[16];
};

struct alignas(16) DrawPushConstants {
	ModelViewProj mvp;
	uint32_t	  primitiveCount	   = 0;
	uint32_t	  verticesPerPrimitive = 3;
	uint32_t	  padding[2]		   = {0, 0};
};

static std::vector<uint32_t> readSpirvFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) throw std::runtime_error("Failed to open shader file: " + filename);

	const size_t fileSize = static_cast<size_t>(file.tellg());
	if (fileSize % sizeof(uint32_t) != 0) throw std::runtime_error("Invalid SPIR-V size: " + filename);

	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
	return buffer;
}

static void matrixIdentity(float* m) {
	for (int i = 0; i < 16; ++i)
		m[i] = 0.0f;
	m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void matrixPerspective(float* m, float fov, float aspect, float zNear, float zFar) {
	const float tanHalfFov = std::tan(fov * 0.5f);
	for (int i = 0; i < 16; ++i)
		m[i] = 0.0f;
	m[0]  = 1.0f / (aspect * tanHalfFov);
	m[5]  = 1.0f / tanHalfFov;
	m[10] = -(zFar + zNear) / (zFar - zNear);
	m[11] = -1.0f;
	m[14] = -(2.0f * zFar * zNear) / (zFar - zNear);
}

static void matrixRotationY(float* m, float a) {
	matrixIdentity(m);
	const float c = std::cos(a);
	const float s = std::sin(a);
	m[0]		  = c;
	m[2]		  = s;
	m[8]		  = -s;
	m[10]		  = c;
}

static void matrixTranslation(float* m, float x, float y, float z) {
	matrixIdentity(m);
	m[12] = x;
	m[13] = y;
	m[14] = z;
}

struct SdlGuard {
	SdlGuard() {
		if (SDL_Init(SDL_INIT_VIDEO) != 0) {
			throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
		}
	}
	~SdlGuard() { SDL_Quit(); }
};

struct WindowDeleter {
	void operator()(SDL_Window* w) const {
		if (w) SDL_DestroyWindow(w);
	}
};

static bool loadMeshOrFallback(const std::string& objPath, std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices) {
	objLoader::Mesh mesh;
	if (objLoader::loadOBJ(objPath, mesh)) {
		outVertices.reserve(mesh.vertices.size());
		for (const auto& v : mesh.vertices) {
			Vertex vv{};
			vv.pos[0]	   = v.pos[0];
			vv.pos[1]	   = v.pos[1];
			vv.pos[2]	   = v.pos[2];
			vv.color[0]	   = v.color[0];
			vv.color[1]	   = v.color[1];
			vv.color[2]	   = v.color[2];
			vv.texCoord[0] = v.texCoord[0];
			vv.texCoord[1] = v.texCoord[1];
			outVertices.push_back(vv);
		}
		outIndices = mesh.indices;
		return true;
	}

	outVertices = {
			{{-0.5f, -0.5f, 0.5f}, {1, 0, 0}, {0, 0}},	{{0.5f, -0.5f, 0.5f}, {1, 0, 0}, {1, 0}},  {{0.5f, 0.5f, 0.5f}, {1, 0, 0}, {1, 1}},
			{{-0.5f, 0.5f, 0.5f}, {1, 0, 0}, {0, 1}},	{{0.5f, -0.5f, 0.5f}, {0, 1, 0}, {0, 0}},  {{0.5f, -0.5f, -0.5f}, {0, 1, 0}, {1, 0}},
			{{0.5f, 0.5f, -0.5f}, {0, 1, 0}, {1, 1}},	{{0.5f, 0.5f, 0.5f}, {0, 1, 0}, {0, 1}},   {{0.5f, -0.5f, -0.5f}, {0, 0, 1}, {0, 0}},
			{{-0.5f, -0.5f, -0.5f}, {0, 0, 1}, {1, 0}}, {{-0.5f, 0.5f, -0.5f}, {0, 0, 1}, {1, 1}}, {{0.5f, 0.5f, -0.5f}, {0, 0, 1}, {0, 1}},
			{{-0.5f, -0.5f, -0.5f}, {1, 1, 0}, {0, 0}}, {{-0.5f, -0.5f, 0.5f}, {1, 1, 0}, {1, 0}}, {{-0.5f, 0.5f, 0.5f}, {1, 1, 0}, {1, 1}},
			{{-0.5f, 0.5f, -0.5f}, {1, 1, 0}, {0, 1}},	{{-0.5f, 0.5f, 0.5f}, {1, 0, 1}, {0, 0}},  {{0.5f, 0.5f, 0.5f}, {1, 0, 1}, {1, 0}},
			{{0.5f, 0.5f, -0.5f}, {1, 0, 1}, {1, 1}},	{{-0.5f, 0.5f, -0.5f}, {1, 0, 1}, {0, 1}}, {{-0.5f, -0.5f, -0.5f}, {0, 1, 1}, {0, 0}},
			{{0.5f, -0.5f, -0.5f}, {0, 1, 1}, {1, 0}},	{{0.5f, -0.5f, 0.5f}, {0, 1, 1}, {1, 1}},  {{-0.5f, -0.5f, 0.5f}, {0, 1, 1}, {0, 1}},
	};
	outIndices = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};
	return false;
}

constexpr uint32_t	  kMeshWorkgroupSize	 = 32;
constexpr const char* kDefaultMeshShaderPath = "shaders/textured.mesh.spv";

int main(int argc, char* argv[]) {
	try {
		// constexpr int   kWidth  = 800;
		// constexpr int   kHeight = 600;
		constexpr int	kWidth			= 3840;
		constexpr int	kHeight			= 2160;
		constexpr float kFov			= 3.14159f / 4.0f;
		constexpr float kNear			= 0.1f;
		constexpr float kFar			= 1000.0f;
		constexpr float kRotSpeed		= 0.8f;
		constexpr float kCameraDistance = 5.0f;

		std::cout << "=== Render API Cube Demo ===\n";

		SdlGuard sdl;

		std::unique_ptr<SDL_Window, WindowDeleter> window(SDL_CreateWindow(
				"Render API - Cube Demo", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, kWidth, kHeight, SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN));
		if (!window) throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());

		unsigned int extensionCount = 0;
		SDL_Vulkan_GetInstanceExtensions(window.get(), &extensionCount, nullptr);
		std::vector<const char*> extensions(extensionCount);
		SDL_Vulkan_GetInstanceExtensions(window.get(), &extensionCount, extensions.data());

		INIT_RENDER_API;

		renderApi::instance::Config instanceConfig;
		instanceConfig			  = renderApi::instance::Config::ReleaseDefault("CubeDemo");
		instanceConfig.extensions = extensions;

		const auto instanceInit = renderApi::initNewInstance(instanceConfig);
		if (instanceInit != renderApi::instance::INIT_VK_INSTANCE_SUCCESS) {
			throw std::runtime_error("Vulkan instance init failed: " + std::to_string(instanceInit));
		}

		auto* instance = renderApi::getInstance(0);
		if (!instance) throw std::runtime_error("Instance not found");

		renderApi::device::Config gpuConfig{};
		gpuConfig.graphics = 1;
		gpuConfig.compute  = 0;
		gpuConfig.transfer = 0;

		const auto deviceInit = instance->addGPU(gpuConfig);
		if (deviceInit != renderApi::device::INIT_DEVICE_SUCCESS) {
			throw std::runtime_error("GPU init failed: " + std::to_string(deviceInit));
		}

		auto* gpu = instance->getGPU(0);
		if (!gpu) throw std::runtime_error("GPU not found");

		std::cout << "GPU: " << gpu->name << "\n";

		const std::string objPath			 = (argc > 1) ? argv[1] : "models/cube.obj";
		bool			  meshShadersEnabled = gpu->meshShaderSupported;
		std::cout << "Mesh shaders: " << (meshShadersEnabled ? "enabled" : "disabled") << "\n";
		if (meshShadersEnabled) {
			std::cout << "  Mesh shader module: " << kDefaultMeshShaderPath << "\n";
		}

		std::vector<Vertex>	  vertices;
		std::vector<uint32_t> indices;
		loadMeshOrFallback(objPath, vertices, indices);

		auto vertexBuffer = renderApi::createVertexBuffer(gpu, vertices);
		auto indexBuffer  = renderApi::createIndexBuffer(gpu, indices);
		if (!vertexBuffer.isValid() || !indexBuffer.isValid()) throw std::runtime_error("Buffer creation failed");

		renderApi::Buffer meshVertexStorage;
		renderApi::Buffer meshIndexStorage;
		uint32_t		  primitiveCount	= static_cast<uint32_t>(indices.size() / 3);
		uint32_t		  meshDispatchCount = 0;

		if (meshShadersEnabled) {
			if (primitiveCount == 0) {
				std::cout << "Mesh shaders disabled: primitive count is zero.\n";
				meshShadersEnabled = false;
			} else {
				const size_t vertexBytes = vertices.size() * sizeof(Vertex);
				const size_t indexBytes	 = indices.size() * sizeof(uint32_t);
				meshVertexStorage		 = renderApi::createStorageBuffer(gpu, vertexBytes);
				meshIndexStorage		 = renderApi::createStorageBuffer(gpu, indexBytes);
				if (!meshVertexStorage.isValid() || !meshIndexStorage.isValid()) {
					throw std::runtime_error("Failed to allocate mesh shader storage buffers");
				}
				meshVertexStorage.upload(vertices.data(), vertexBytes);
				meshIndexStorage.upload(indices.data(), indexBytes);
				meshDispatchCount = (primitiveCount + kMeshWorkgroupSize - 1) / kMeshWorkgroupSize;
				if (meshDispatchCount == 0) meshDispatchCount = 1;
			}
		}

		constexpr uint32_t	 texW = 256, texH = 256;
		std::vector<uint8_t> textureData(texW * texH * 4);
		for (uint32_t y = 0; y < texH; ++y) {
			for (uint32_t x = 0; x < texW; ++x) {
				const uint32_t i	 = (y * texW + x) * 4;
				const bool	   white = (((x / 32) + (y / 32)) & 1u) == 0u;
				const uint8_t  c	 = white ? 255 : 64;
				textureData[i + 0]	 = c;
				textureData[i + 1]	 = c;
				textureData[i + 2]	 = c;
				textureData[i + 3]	 = 255;
			}
		}

		auto texture = renderApi::createTexture2D(gpu, texW, texH, VK_FORMAT_R8G8B8A8_UNORM, textureData.data(), textureData.size(), false);
		if (!texture.isValid()) throw std::runtime_error("Texture creation failed");

		renderApi::gpuTask::GpuTask task("CubeRender", gpu);

		auto* pipeline = task.createGraphicsPipeline("CubePipeline");
		if (!pipeline) throw std::runtime_error("Pipeline creation failed");

		pipeline->setOutputTarget(renderApi::gpuTask::OutputTarget::SDL_SURFACE);
		pipeline->setSDLWindow(window.get());
		pipeline->setPresentMode(VK_PRESENT_MODE_MAILBOX_KHR);
		pipeline->setSwapchainImageCount(2);

		if (meshShadersEnabled) {
			pipeline->setMeshShader(readSpirvFile(kDefaultMeshShaderPath));
		} else {
			pipeline->setVertexShader(readSpirvFile("shaders/textured.vert.spv"));
		}
		pipeline->setFragmentShader(readSpirvFile("shaders/textured.frag.spv"));

		if (!meshShadersEnabled) {
			pipeline->addVertexBinding(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);
			pipeline->addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos));
			pipeline->addVertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color));
			pipeline->addVertexAttribute(2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord));
		}

		pipeline->setViewport(kWidth, kHeight);
		pipeline->setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
		pipeline->setDepthStencil(true, true, VK_COMPARE_OP_LESS);
		pipeline->setColorBlendAttachment(false);
		VkShaderStageFlags pushConstantStages = VK_SHADER_STAGE_VERTEX_BIT;
		if (meshShadersEnabled) {
			pushConstantStages |= VK_SHADER_STAGE_MESH_BIT_EXT;
		}
		pipeline->addPushConstantRange(pushConstantStages, 0, sizeof(DrawPushConstants));

		task.enableDescriptorManager(true);
		auto* descriptorManager = task.getDescriptorManager();
		auto* descriptorSet		= descriptorManager->createSet(0);
		if (meshShadersEnabled) {
			descriptorSet->addBuffer(1, &meshVertexStorage, renderApi::descriptor::DescriptorType::STORAGE_BUFFER, VK_SHADER_STAGE_MESH_BIT_EXT);
			descriptorSet->addBuffer(2, &meshIndexStorage, renderApi::descriptor::DescriptorType::STORAGE_BUFFER, VK_SHADER_STAGE_MESH_BIT_EXT);
		}
		descriptorSet->addTexture(0, &texture, VK_SHADER_STAGE_FRAGMENT_BIT);

		task.addVertexBuffer(&vertexBuffer);
		task.setIndexBuffer(&indexBuffer, VK_INDEX_TYPE_UINT32);
		task.setIndexedDrawParams(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
		if (meshShadersEnabled) {
			task.setMeshTaskCount(meshDispatchCount, 1, 1);
		}

		if (!task.build(kWidth, kHeight)) throw std::runtime_error("GpuTask build failed");

		bool running = true;
		bool paused	 = false;

		float angle = 0.0f;

		ModelViewProj	  mvp{};
		DrawPushConstants drawConstants{};
		drawConstants.primitiveCount	   = primitiveCount;
		drawConstants.verticesPerPrimitive = 3;
		matrixPerspective(mvp.proj, kFov, float(kWidth) / float(kHeight), kNear, kFar);
		matrixTranslation(mvp.view, 0.0f, 0.0f, -kCameraDistance);

		using clock = std::chrono::high_resolution_clock;
		auto prev	= clock::now();
		auto fpsT0	= prev;
		int	 frames = 0;

		SDL_Event e{};
		while (running) {
			while (SDL_PollEvent(&e)) {
				switch (e.type) {
				case SDL_QUIT:
					running = false;
					break;
				case SDL_KEYDOWN:
					if (e.key.keysym.sym == SDLK_ESCAPE) {
						running = false;
					} else if (e.key.keysym.sym == SDLK_p) {
						paused = !paused;
					}
					break;
				default:
					break;
				}
			}

			const auto	now = clock::now();
			const float dt	= std::chrono::duration<float>(now - prev).count();
			prev			= now;

			if (!paused) angle += dt * kRotSpeed;

			matrixRotationY(mvp.model, angle);

			drawConstants.mvp			  = mvp;
			VkShaderStageFlags pushStages = VK_SHADER_STAGE_VERTEX_BIT;
			if (meshShadersEnabled) {
				pushStages |= VK_SHADER_STAGE_MESH_BIT_EXT;
			}
			task.pushConstants(pushStages, 0, sizeof(DrawPushConstants), &drawConstants);
			task.execute();

			++frames;
			const float fpsElapsed = std::chrono::duration<float>(now - fpsT0).count();
			if (fpsElapsed >= 1.0f) {
				const float fps = frames / fpsElapsed;
				std::cout << "FPS: " << int(fps) << "\n";
				frames = 0;
				fpsT0  = now;
			}
		}

		vkDeviceWaitIdle(gpu->device);

		task.destroy();
		vertexBuffer.destroy();
		indexBuffer.destroy();
		meshVertexStorage.destroy();
		meshIndexStorage.destroy();

		std::cout << "Done.\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Fatal: " << ex.what() << "\n";
		return 1;
	}
}
