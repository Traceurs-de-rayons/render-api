#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>

namespace imageLoader {

struct ImageData {
    std::vector<uint8_t> pixels;
    int width;
    int height;
    int channels;
    bool valid;
};

inline ImageData loadImage(const std::string& filepath, int desiredChannels = 4) {
    ImageData result;
    result.valid = false;
    result.width = 0;
    result.height = 0;
    result.channels = 0;

    // Load image using stb_image
    int width, height, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, desiredChannels);

    if (!data) {
        std::cerr << "Failed to load image: " << filepath << std::endl;
        std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
        return result;
    }

    // If we forced a channel count, use that
    if (desiredChannels != 0) {
        channels = desiredChannels;
    }

    // Copy data to vector
    size_t dataSize = width * height * channels;
    result.pixels.resize(dataSize);
    memcpy(result.pixels.data(), data, dataSize);

    // Free stb_image data
    stbi_image_free(data);

    // Fill result
    result.width = width;
    result.height = height;
    result.channels = channels;
    result.valid = true;

    std::cout << "Image loaded: " << filepath << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Size: " << dataSize << " bytes" << std::endl;

    return result;
}

// Helper to create a procedural checkerboard texture
inline ImageData createCheckerboard(int width, int height, int squareSize = 32) {
    ImageData result;
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.valid = true;
    result.pixels.resize(width * height * 4);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4;
            bool isWhite = ((x / squareSize) + (y / squareSize)) % 2 == 0;
            uint8_t color = isWhite ? 255 : 64;
            result.pixels[idx + 0] = color;  // R
            result.pixels[idx + 1] = color;  // G
            result.pixels[idx + 2] = color;  // B
            result.pixels[idx + 3] = 255;    // A
        }
    }

    std::cout << "Created procedural checkerboard: " << width << "x" << height << std::endl;
    return result;
}

// Helper to create a solid color texture
inline ImageData createSolidColor(int width, int height, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    ImageData result;
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.valid = true;
    result.pixels.resize(width * height * 4);

    for (int i = 0; i < width * height; i++) {
        result.pixels[i * 4 + 0] = r;
        result.pixels[i * 4 + 1] = g;
        result.pixels[i * 4 + 2] = b;
        result.pixels[i * 4 + 3] = a;
    }

    std::cout << "Created solid color texture: " << width << "x" << height 
              << " RGBA(" << (int)r << "," << (int)g << "," << (int)b << "," << (int)a << ")" << std::endl;
    return result;
}

// Helper to create a gradient texture
inline ImageData createGradient(int width, int height, bool horizontal = true) {
    ImageData result;
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.valid = true;
    result.pixels.resize(width * height * 4);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4;
            uint8_t value = horizontal ? (x * 255 / width) : (y * 255 / height);
            result.pixels[idx + 0] = value;
            result.pixels[idx + 1] = value;
            result.pixels[idx + 2] = value;
            result.pixels[idx + 3] = 255;
        }
    }

    std::cout << "Created gradient texture: " << width << "x" << height 
              << (horizontal ? " (horizontal)" : " (vertical)") << std::endl;
    return result;
}

} // namespace imageLoader