cmake_minimum_required(VERSION 3.16)
project(render-api VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
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

option(BUILD_RENDER_API_EXAMPLES "Build examples for render-api" ON)

if(BUILD_RENDER_API_EXAMPLES)
	add_executable(render-api-example examples/main.cpp)
	target_link_libraries(render-api-example PRIVATE render-api)
	target_compile_features(render-api-example PRIVATE cxx_std_17)
endif()
