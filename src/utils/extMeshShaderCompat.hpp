#ifndef RENDER_API_EXT_MESH_SHADER_COMPAT_HPP
#define RENDER_API_EXT_MESH_SHADER_COMPAT_HPP

#include <vulkan/vulkan_core.h>

#ifndef VK_EXT_mesh_shader
#define VK_EXT_mesh_shader 1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT ((VkStructureType)1000322000)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT ((VkStructureType)1000322001)
#define VK_SHADER_STAGE_TASK_BIT_EXT ((VkShaderStageFlagBits)0x00000040)
#define VK_SHADER_STAGE_MESH_BIT_EXT ((VkShaderStageFlagBits)0x00000080)
#define VK_EXT_MESH_SHADER_EXTENSION_NAME "VK_EXT_mesh_shader"

typedef struct VkPhysicalDeviceMeshShaderFeaturesEXT {
	VkStructureType sType;
	void*			 pNext;
	VkBool32		 taskShader;
	VkBool32		 meshShader;
	VkBool32		 multiviewMeshShader;
	VkBool32		 primitiveFragmentShadingRateMeshShader;
	VkBool32		 meshShaderQueries;
} VkPhysicalDeviceMeshShaderFeaturesEXT;
#endif

#ifndef VK_QUEUE_VIDEO_DECODE_BIT_KHR
#define VK_QUEUE_VIDEO_DECODE_BIT_KHR ((VkQueueFlagBits)0x00000020)
#endif

#ifndef VK_QUEUE_VIDEO_ENCODE_BIT_KHR
#define VK_QUEUE_VIDEO_ENCODE_BIT_KHR ((VkQueueFlagBits)0x00000040)
#endif

#ifndef VK_QUEUE_OPTICAL_FLOW_BIT_NV
#define VK_QUEUE_OPTICAL_FLOW_BIT_NV ((VkQueueFlagBits)0x00000100)
#endif

#endif