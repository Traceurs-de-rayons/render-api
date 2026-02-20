#include "utils.hpp"

#include "buffer/buffer.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>

bool saveBufferAsPPM(const std::string& filename, renderApi::Buffer& buffer, uint32_t width, uint32_t height) {
	if (!buffer.isValid()) {
		std::cerr << "Cannot save PPM: invalid buffer" << std::endl;
		return false;
	}

	size_t expectedSize = width * height * 4;
	if (buffer.getSize() < expectedSize) {
		std::cerr << "Cannot save PPM: buffer size mismatch (expected " << expectedSize << ", got " << buffer.getSize() << ")" << std::endl;
		return false;
	}

	void* data = buffer.map();
	if (!data) {
		std::cerr << "Cannot save PPM: failed to map buffer" << std::endl;
		return false;
	}

	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Cannot save PPM: failed to open file " << filename << std::endl;
		buffer.unmap();
		return false;
	}

	file << "P6\n" << width << " " << height << "\n255\n";

	uint8_t* pixels = static_cast<uint8_t*>(data);
	for (uint32_t y = 0; y < height; ++y) {
		for (uint32_t x = 0; x < width; ++x) {
			uint32_t idx = (y * width + x) * 4;
			file << pixels[idx + 0];
			file << pixels[idx + 1];
			file << pixels[idx + 2];
		}
	}

	file.close();
	buffer.unmap();

	std::cout << "Image saved to " << filename << " (" << width << "x" << height << ")" << std::endl;
	return true;
}