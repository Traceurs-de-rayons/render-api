# Makefile for render-api examples
# This allows building examples directly from the examples directory

.PHONY: all clean build run help

# Default target
all: build

# Build the examples
build:
	@echo "=== Building render-api examples ==="
	@cd .. && mkdir -p build && cd build && cmake -DBUILD_RENDER_API_EXAMPLES=ON .. && make

# Clean build directory
clean:
	@echo "=== Cleaning build directory ==="
	@cd .. && rm -rf build/*

# Rebuild from scratch
rebuild: clean build

# Run the main example
run:
	@echo "=== Running cube example ==="
	@cd ../build && ./render-api-example

# Build and run
go: build run

# Compile shaders manually
shaders:
	@echo "=== Compiling shaders ==="
	@cd .. && glslc shaders/cube.vert -o shaders/cube.vert.spv
	@cd .. && glslc shaders/cube.frag -o shaders/cube.frag.spv
	@echo "Shaders compiled successfully!"

# Show help
help:
	@echo "Render API Examples - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build the examples"
	@echo "  make build    - Build the examples"
	@echo "  make clean    - Clean the build directory"
	@echo "  make rebuild  - Clean and rebuild"
	@echo "  make run      - Run the cube example"
	@echo "  make go       - Build and run"
	@echo "  make shaders  - Recompile shaders"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make go       # Quick build and run"
	@echo "  make rebuild  # Full rebuild from scratch"
