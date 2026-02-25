#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

namespace objLoader {

struct Vertex {
    float pos[3];
    float normal[3];
    float texCoord[2];
    float color[3]; // Default color if not specified
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

inline bool loadOBJ(const std::string& filepath, Mesh& mesh, bool generateNormals = true) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filepath << std::endl;
        return false;
    }

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texCoords;
    
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    // Map to avoid duplicate vertices
    std::unordered_map<std::string, uint32_t> uniqueVertices;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            // Vertex position
            float x, y, z;
            iss >> x >> y >> z;
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
        }
        else if (prefix == "vn") {
            // Vertex normal
            float x, y, z;
            iss >> x >> y >> z;
            normals.push_back(x);
            normals.push_back(y);
            normals.push_back(z);
        }
        else if (prefix == "vt") {
            // Texture coordinate
            float u, v;
            iss >> u >> v;
            texCoords.push_back(u);
            texCoords.push_back(v);
        }
        else if (prefix == "f") {
            // Face
            std::string vertex1, vertex2, vertex3;
            iss >> vertex1 >> vertex2 >> vertex3;
            
            std::vector<std::string> faceVertices = {vertex1, vertex2, vertex3};
            
            for (const auto& vertexStr : faceVertices) {
                // Check if we've seen this vertex before
                if (uniqueVertices.find(vertexStr) != uniqueVertices.end()) {
                    indices.push_back(uniqueVertices[vertexStr]);
                    continue;
                }
                
                // Parse vertex string (format: v/vt/vn or v//vn or v/vt or v)
                Vertex vertex = {};
                
                // Default color (white)
                vertex.color[0] = 1.0f;
                vertex.color[1] = 1.0f;
                vertex.color[2] = 1.0f;
                
                size_t pos1 = vertexStr.find('/');
                size_t pos2 = vertexStr.rfind('/');
                
                // Position index
                int posIdx = std::stoi(vertexStr.substr(0, pos1)) - 1;
                if (posIdx >= 0 && posIdx * 3 + 2 < positions.size()) {
                    vertex.pos[0] = positions[posIdx * 3];
                    vertex.pos[1] = positions[posIdx * 3 + 1];
                    vertex.pos[2] = positions[posIdx * 3 + 2];
                }
                
                // Texture coordinate index
                if (pos1 != std::string::npos && pos2 != std::string::npos && pos1 != pos2) {
                    std::string texStr = vertexStr.substr(pos1 + 1, pos2 - pos1 - 1);
                    if (!texStr.empty()) {
                        int texIdx = std::stoi(texStr) - 1;
                        if (texIdx >= 0 && texIdx * 2 + 1 < texCoords.size()) {
                            vertex.texCoord[0] = texCoords[texIdx * 2];
                            vertex.texCoord[1] = texCoords[texIdx * 2 + 1];
                        }
                    }
                }
                
                // Normal index
                if (pos2 != std::string::npos && pos2 + 1 < vertexStr.length()) {
                    int normIdx = std::stoi(vertexStr.substr(pos2 + 1)) - 1;
                    if (normIdx >= 0 && normIdx * 3 + 2 < normals.size()) {
                        vertex.normal[0] = normals[normIdx * 3];
                        vertex.normal[1] = normals[normIdx * 3 + 1];
                        vertex.normal[2] = normals[normIdx * 3 + 2];
                    }
                }
                
                uint32_t index = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
                indices.push_back(index);
                uniqueVertices[vertexStr] = index;
            }
        }
    }

    file.close();

    // Generate normals if none were provided and requested
    if (generateNormals && normals.empty()) {
        std::cout << "Generating normals for " << filepath << std::endl;
        
        // Initialize all normals to zero
        for (auto& vertex : vertices) {
            vertex.normal[0] = 0.0f;
            vertex.normal[1] = 0.0f;
            vertex.normal[2] = 0.0f;
        }
        
        // Calculate face normals and accumulate
        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t idx0 = indices[i];
            uint32_t idx1 = indices[i + 1];
            uint32_t idx2 = indices[i + 2];
            
            Vertex& v0 = vertices[idx0];
            Vertex& v1 = vertices[idx1];
            Vertex& v2 = vertices[idx2];
            
            // Calculate edge vectors
            float edge1[3] = {
                v1.pos[0] - v0.pos[0],
                v1.pos[1] - v0.pos[1],
                v1.pos[2] - v0.pos[2]
            };
            
            float edge2[3] = {
                v2.pos[0] - v0.pos[0],
                v2.pos[1] - v0.pos[1],
                v2.pos[2] - v0.pos[2]
            };
            
            // Calculate cross product (face normal)
            float normal[3] = {
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            };
            
            // Accumulate to vertex normals
            v0.normal[0] += normal[0];
            v0.normal[1] += normal[1];
            v0.normal[2] += normal[2];
            
            v1.normal[0] += normal[0];
            v1.normal[1] += normal[1];
            v1.normal[2] += normal[2];
            
            v2.normal[0] += normal[0];
            v2.normal[1] += normal[1];
            v2.normal[2] += normal[2];
        }
        
        // Normalize all vertex normals
        for (auto& vertex : vertices) {
            float length = std::sqrt(
                vertex.normal[0] * vertex.normal[0] +
                vertex.normal[1] * vertex.normal[1] +
                vertex.normal[2] * vertex.normal[2]
            );
            
            if (length > 0.0001f) {
                vertex.normal[0] /= length;
                vertex.normal[1] /= length;
                vertex.normal[2] /= length;
            }
        }
    }

    mesh.vertices = std::move(vertices);
    mesh.indices = std::move(indices);

    std::cout << "Loaded OBJ: " << filepath << std::endl;
    std::cout << "  Vertices: " << mesh.vertices.size() << std::endl;
    std::cout << "  Indices: " << mesh.indices.size() << std::endl;
    std::cout << "  Triangles: " << mesh.indices.size() / 3 << std::endl;

    return true;
}

} // namespace objLoader