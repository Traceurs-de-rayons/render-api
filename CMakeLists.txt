cmake_minimum_required(VERSION 3.16)
project(render-api VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom Vulkan SDK path - use environment variable if available, otherwise use default
if(DEFINED ENV{VULKAN_SDK})
	set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}" CACHE PATH "Path to Vulkan SDK")
else()
	set(VULKAN_SDK_PATH "/home/madelvin/goinfre/vulkansdk-linux-x86_64-1.4.341.1/vulkan-1.4.341.1" CACHE PATH "Path to Vulkan SDK")
endif()

# Configure Vulkan manually to bypass system installation
set(VULKAN_INCLUDE_DIR "${VULKAN_SDK_PATH}/x86_64/include")
set(VULKAN_LIBRARY "${VULKAN_SDK_PATH}/x86_64/lib/libvulkan.so")
set(VULKAN_LAYER_PATH "${VULKAN_SDK_PATH}/x86_64/share/vulkan/explicit_layer.d")

if(NOT EXISTS ${VULKAN_INCLUDE_DIR})
	message(FATAL_ERROR "Vulkan SDK not found at: ${VULKAN_SDK_PATH}\nPlease update VULKAN_SDK_PATH in CMakeLists.txt")
endif()

message(STATUS "Using custom Vulkan SDK: ${VULKAN_SDK_PATH}")
message(STATUS "Vulkan include dir: ${VULKAN_INCLUDE_DIR}")
message(STATUS "Vulkan library: ${VULKAN_LIBRARY}")
message(STATUS "Vulkan layers: ${VULKAN_LAYER_PATH}")

# Create Vulkan imported target manually
add_library(Vulkan::Vulkan SHARED IMPORTED)
set_target_properties(Vulkan::Vulkan PROPERTIES
	IMPORTED_LOCATION "${VULKAN_LIBRARY}"
	INTERFACE_INCLUDE_DIRECTORIES "${VULKAN_INCLUDE_DIR}"
)

find_package(SDL2 REQUIRED)

file(GLOB_RECURSE RENDER_API_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_library(render-api STATIC ${RENDER_API_SOURCES})
target_link_libraries(render-api PUBLIC Vulkan::Vulkan SDL2::SDL2)

file(GLOB SRC_CHILDREN RELATIVE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/*)
set(SRC_INCLUDE_DIRS)
foreach(CHILD ${SRC_CHILDREN})
	if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/src/${CHILD})
		list(APPEND SRC_INCLUDE_DIRS "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/${CHILD}>")
	endif()
endforeach()

target_include_directories(render-api
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
		${SRC_INCLUDE_DIRS}
		$<INSTALL_INTERFACE:include>
)

target_compile_features(render-api PUBLIC cxx_std_17)
set_target_properties(render-api PROPERTIES OUTPUT_NAME "render-api")

include(GNUInstallDirs)
install(TARGETS render-api
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Shader build and installation - copy precompiled .spv shaders
if(EXISTS ${CMAKE_SOURCE_DIR}/shaders)
	add_custom_target(shaders ALL
		COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_SOURCE_DIR}/shaders
		${CMAKE_BINARY_DIR}/shaders
		COMMENT "Copying shaders to build directory"
	)

	add_custom_target(shaders_update
		COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_SOURCE_DIR}/shaders
		${CMAKE_BINARY_DIR}/shaders
		COMMENT "Updating shaders in build directory"
	)

	install(DIRECTORY ${CMAKE_SOURCE_DIR}/shaders/ DESTINATION shaders FILES_MATCHING PATTERN "*.spv")
else()
	message(WARNING "shaders directory not found, skipping shader copy")
endif()

option(BUILD_RENDER_API_EXAMPLES "Build examples for render-api" OFF)

if(BUILD_RENDER_API_EXAMPLES)
	if(EXISTS ${CMAKE_SOURCE_DIR}/examples/main.cpp)
		add_executable(render-api-example examples/main.cpp)
		target_link_libraries(render-api-example PRIVATE render-api)
		target_compile_features(render-api-example PRIVATE cxx_std_17)

		# Set RPATH to include Vulkan SDK lib directory
		set_target_properties(render-api-example PROPERTIES
			BUILD_RPATH "${VULKAN_SDK_PATH}/x86_64/lib"
			INSTALL_RPATH "${VULKAN_SDK_PATH}/x86_64/lib"
			BUILD_WITH_INSTALL_RPATH FALSE
			SKIP_BUILD_RPATH FALSE
		)
	else()
		message(WARNING "examples/main.cpp not found, skipping render-api-example")
	endif()

	if(EXISTS ${CMAKE_SOURCE_DIR}/examples/cube_demo.cpp)
		add_executable(cube-demo examples/cube_demo.cpp)
		target_link_libraries(cube-demo PRIVATE render-api)
		target_compile_features(cube-demo PRIVATE cxx_std_17)

		# Set RPATH to include Vulkan SDK lib directory
		set_target_properties(cube-demo PROPERTIES
			BUILD_RPATH "${VULKAN_SDK_PATH}/x86_64/lib"
			INSTALL_RPATH "${VULKAN_SDK_PATH}/x86_64/lib"
			BUILD_WITH_INSTALL_RPATH FALSE
			SKIP_BUILD_RPATH FALSE
		)
	else()
		message(WARNING "examples/cube_demo.cpp not found, skipping cube-demo")
	endif()
endif()
