cmake_minimum_required(VERSION 3.16)
project(render-api VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Vulkan SDK
if(DEFINED ENV{VULKAN_SDK})
	set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
	# Check if path already includes x86_64
	if(EXISTS "${VULKAN_SDK_PATH}/include/vulkan/vulkan.h")
		set(VULKAN_ARCH_PATH "${VULKAN_SDK_PATH}")
	else()
		set(VULKAN_ARCH_PATH "${VULKAN_SDK_PATH}/x86_64")
	endif()
else()
	message(FATAL_ERROR "VULKAN_SDK environment variable not set")
endif()

# Configure Vulkan
set(VULKAN_INCLUDE_DIR "${VULKAN_ARCH_PATH}/include")
set(VULKAN_LIBRARY "${VULKAN_ARCH_PATH}/lib/libvulkan.so")

if(NOT EXISTS ${VULKAN_INCLUDE_DIR}/vulkan/vulkan.h)
	message(FATAL_ERROR "Vulkan SDK not found at: ${VULKAN_SDK_PATH}")
endif()

# Create Vulkan target
add_library(Vulkan::Vulkan SHARED IMPORTED)
set_target_properties(Vulkan::Vulkan PROPERTIES
	IMPORTED_LOCATION "${VULKAN_LIBRARY}"
	INTERFACE_INCLUDE_DIRECTORIES "${VULKAN_INCLUDE_DIR}"
)

# Find SDL2
find_package(SDL2 REQUIRED)

# Collect sources
file(GLOB_RECURSE RENDER_API_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create library
add_library(render-api STATIC ${RENDER_API_SOURCES})
target_link_libraries(render-api PUBLIC Vulkan::Vulkan SDL2::SDL2)

# Add all src subdirectories to includes
file(GLOB SRC_SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
set(SRC_INCLUDE_DIRS)
foreach(SUBDIR ${SRC_SUBDIRS})
	if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/${SUBDIR})
		list(APPEND SRC_INCLUDE_DIRS "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/${SUBDIR}>")
	endif()
endforeach()

target_include_directories(render-api
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
		${SRC_INCLUDE_DIRS}
		$<INSTALL_INTERFACE:include>
)

target_compile_features(render-api PUBLIC cxx_std_17)
