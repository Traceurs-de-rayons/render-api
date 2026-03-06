cmake_minimum_required(VERSION 3.16)
project(render-api VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom Vulkan SDK path resolution
set(VULKAN_SDK_PATH "" CACHE PATH "Path to Vulkan SDK")

function(_resolve_vulkan_sdk candidate_var)
	set(_result "")
	foreach(_candidate IN LISTS ARGN)
		if(NOT _candidate)
			continue()
		endif()
		file(TO_CMAKE_PATH "${_candidate}" _candidate_normalized)
		if(NOT IS_DIRECTORY "${_candidate_normalized}")
			continue()
		endif()
		if(EXISTS "${_candidate_normalized}/x86_64/include/vulkan/vulkan.h")
			set(_result "${_candidate_normalized}")
			break()
		endif()
		file(GLOB _child_versions LIST_DIRECTORIES true "${_candidate_normalized}/*")
		list(SORT _child_versions)
		list(REVERSE _child_versions)
		foreach(_child IN LISTS _child_versions)
			if(EXISTS "${_child}/x86_64/include/vulkan/vulkan.h")
				set(_result "${_child}")
				break()
			endif()
		endforeach()
		if(_result)
			break()
		endif()
	endforeach()
	set(${candidate_var} "${_result}" PARENT_SCOPE)
endfunction()

set(_vulkan_sdk_ready OFF)
if(VULKAN_SDK_PATH AND EXISTS "${VULKAN_SDK_PATH}/x86_64/include/vulkan/vulkan.h")
	set(_vulkan_sdk_ready ON)
endif()

if(NOT _vulkan_sdk_ready)
	set(_vulkan_candidates)
	if(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
		list(APPEND _vulkan_candidates "$ENV{VULKAN_SDK}")
	endif()
	list(APPEND _vulkan_candidates
		"/home/madelvin/Documents/tools/vulkan"
		"/home/madelvin/goinfre/vulkansdk-linux-x86_64-1.4.341.1/vulkan-1.4.341.1"
		"/home/madelvin/goinfre"
	)
	list(REMOVE_DUPLICATES _vulkan_candidates)
	_resolve_vulkan_sdk(_detected_vulkan_sdk ${_vulkan_candidates})
	if(_detected_vulkan_sdk)
		set(VULKAN_SDK_PATH "${_detected_vulkan_sdk}" CACHE PATH "Path to Vulkan SDK" FORCE)
		set(_vulkan_sdk_ready ON)
	endif()
endif()

if(NOT _vulkan_sdk_ready)
	message(FATAL_ERROR "Vulkan SDK not found. Set VULKAN_SDK_PATH cache entry or VULKAN_SDK env var to your SDK root (e.g. ~/Documents/tools/vulkan/1.xx.xx).")
endif()

# Configure Vulkan manually to bypass system installation
set(VULKAN_INCLUDE_DIR "${VULKAN_SDK_PATH}/x86_64/include")
set(VULKAN_LIBRARY "${VULKAN_SDK_PATH}/x86_64/lib/libvulkan.so")
set(VULKAN_LAYER_PATH "${VULKAN_SDK_PATH}/x86_64/share/vulkan/explicit_layer.d")

if(NOT EXISTS ${VULKAN_INCLUDE_DIR})
	message(FATAL_ERROR "Vulkan SDK include directory not found at: ${VULKAN_INCLUDE_DIR}\nUpdate VULKAN_SDK_PATH.")
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

file(GLOB_RECURSE RENDER_API_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(render-api STATIC ${RENDER_API_SOURCES})
target_link_libraries(render-api PUBLIC Vulkan::Vulkan SDL2::SDL2)

file(GLOB SRC_CHILDREN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
set(SRC_INCLUDE_DIRS)
foreach(CHILD ${SRC_CHILDREN})
	if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/${CHILD})
		list(APPEND SRC_INCLUDE_DIRS "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/${CHILD}>")
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
set_target_properties(render-api PROPERTIES OUTPUT_NAME "render-api")

include(GNUInstallDirs)
install(TARGETS render-api
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
