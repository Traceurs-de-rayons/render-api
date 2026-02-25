#include "renderApi.hpp"
#include "buffer/buffer.hpp"
#include "gpuTask/gpuTask.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "utils/utils.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <SDL_video.h>
#include <vulkan/vulkan.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include "vulkan/vulkan_core.h"

// Structure pour les vertex avec position et couleur
struct Vertex {
    float pos[3];
    float color[3];
};

// Structure pour les matrices (push constants)
struct ModelViewProj {
    float model[16];
    float view[16];
    float proj[16];
};

// Fonction pour lire un fichier SPIR-V
std::vector<uint32_t> readSpirvFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

// Fonction pour créer une matrice identité
void matrixIdentity(float* mat) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
}

// Fonction pour créer une matrice de perspective
void matrixPerspective(float* mat, float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = 1.0f / (aspect * tanHalfFov);
    mat[5] = 1.0f / tanHalfFov;
    mat[10] = -(far + near) / (far - near);
    mat[11] = -1.0f;
    mat[14] = -(2.0f * far * near) / (far - near);
}

// Fonction pour créer une matrice de rotation Y
void matrixRotationY(float* mat, float angle) {
    matrixIdentity(mat);
    mat[0] = cos(angle);
    mat[2] = sin(angle);
    mat[8] = -sin(angle);
    mat[10] = cos(angle);
}

// Fonction pour créer une matrice de translation
void matrixTranslation(float* mat, float x, float y, float z) {
    matrixIdentity(mat);
    mat[12] = x;
    mat[13] = y;
    mat[14] = z;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Render API Cube Demo ===" << std::endl;

    // Initialiser SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "Erreur SDL_Init: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Créer une fenêtre SDL avec support Vulkan
    SDL_Window* window = SDL_CreateWindow(
        "Render API - Cube Demo",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN
    );

    if (!window) {
        std::cerr << "Erreur création fenêtre: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    std::cout << "Fenêtre SDL créée" << std::endl;

    // Obtenir les extensions Vulkan requises par SDL
    unsigned int extensionCount = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr);
    std::vector<const char*> extensions(extensionCount);
    SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensions.data());

    std::cout << "Extensions SDL Vulkan: " << extensionCount << std::endl;

    // Initialiser Render API
    INIT_RENDER_API;

    // Configuration de l'instance Vulkan
    renderApi::instance::Config instanceConfig;

    #ifdef NDEBUG
        instanceConfig = renderApi::instance::Config::ReleaseDefault("CubeDemo");
    #else
        instanceConfig = renderApi::instance::Config::DebugDefault("CubeDemo");
    #endif

    instanceConfig.extensions = extensions;

    auto result = renderApi::initNewInstance(instanceConfig);
    if (result != renderApi::instance::INIT_VK_INSTANCE_SUCCESS) {
        std::cerr << "Erreur initialisation instance Vulkan: " << result << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::cout << "Instance Vulkan créée" << std::endl;

    // Récupérer l'instance
    auto* instance = renderApi::getInstance(0);
    if (!instance) {
        std::cerr << "Instance Vulkan introuvable" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Ajouter un GPU avec configuration des queues
    // Note: La queue graphics supporte aussi compute et transfer
    renderApi::device::Config gpuConfig;
    gpuConfig.graphics = 1;  // 1 queue graphics (supporte aussi compute/transfer)
    gpuConfig.compute = 0;   // Pas besoin de queue compute dédiée
    gpuConfig.transfer = 0;  // Pas besoin de queue transfer dédiée

    auto deviceResult = instance->addGPU(gpuConfig);
    if (deviceResult != renderApi::device::INIT_DEVICE_SUCCESS) {
        std::cerr << "Erreur initialisation GPU: " << deviceResult << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    auto* gpu = instance->getGPU(0);
    if (!gpu) {
        std::cerr << "GPU introuvable" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::cout << "GPU initialisé: " << gpu->name << std::endl;

    // Créer les vertices du cube avec couleurs différentes par face
    // On duplique les sommets pour avoir des couleurs distinctes par face
    std::vector<Vertex> vertices = {
        // Face avant (rouge) - Z+
        {{-0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},  // 0
        {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},  // 1
        {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},  // 2
        {{-0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},  // 3

        // Face droite (vert) - X+
        {{ 0.5f, -0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},  // 4
        {{ 0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},  // 5
        {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},  // 6
        {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},  // 7

        // Face arrière (bleu) - Z-
        {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},  // 8
        {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},  // 9
        {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},  // 10
        {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},  // 11

        // Face gauche (jaune) - X-
        {{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},  // 12
        {{-0.5f, -0.5f,  0.5f}, {1.0f, 1.0f, 0.0f}},  // 13
        {{-0.5f,  0.5f,  0.5f}, {1.0f, 1.0f, 0.0f}},  // 14
        {{-0.5f,  0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},  // 15

        // Face haut (magenta) - Y+
        {{-0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 1.0f}},  // 16
        {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 1.0f}},  // 17
        {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},  // 18
        {{-0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},  // 19

        // Face bas (cyan) - Y-
        {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}},  // 20
        {{ 0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}},  // 21
        {{ 0.5f, -0.5f,  0.5f}, {0.0f, 1.0f, 1.0f}},  // 22
        {{-0.5f, -0.5f,  0.5f}, {0.0f, 1.0f, 1.0f}},  // 23
    };

    // Indices pour les 6 faces du cube (ordre counter-clockwise vu de l'extérieur)
    std::vector<uint32_t> indices = {
        // Face avant (Z+) - rouge
        0, 1, 2,  2, 3, 0,
        // Face droite (X+) - vert
        4, 5, 6,  6, 7, 4,
        // Face arrière (Z-) - bleu
        8, 9, 10,  10, 11, 8,
        // Face gauche (X-) - jaune
        12, 13, 14,  14, 15, 12,
        // Face haut (Y+) - magenta
        16, 17, 18,  18, 19, 16,
        // Face bas (Y-) - cyan
        20, 21, 22,  22, 23, 20
    };

    std::cout << "Création des buffers..." << std::endl;

    // Créer les buffers
    auto vertexBuffer = renderApi::createVertexBuffer(gpu, vertices);
    auto indexBuffer = renderApi::createIndexBuffer(gpu, indices);

    if (!vertexBuffer.isValid() || !indexBuffer.isValid()) {
        std::cerr << "Erreur création buffers" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::cout << "Buffers créés (vertices: " << vertices.size()
              << ", indices: " << indices.size() << ")" << std::endl;

    // Créer une tâche GPU
    renderApi::gpuTask::GpuTask task("CubeRender", gpu);

    // Créer un pipeline graphique
    auto* pipeline = task.createGraphicsPipeline("CubePipeline");
    if (!pipeline) {
        std::cerr << "Erreur création pipeline" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::cout << "Pipeline graphique créé" << std::endl;

    pipeline->setOutputTarget(renderApi::gpuTask::OutputTarget::SDL_SURFACE);
    pipeline->setSDLWindow(window);

    pipeline->setPresentMode(VK_PRESENT_MODE_IMMEDIATE_KHR);
    pipeline->setSwapchainImageCount(2);

    try {
        std::cout << "Chargement des shaders..." << std::endl;
        auto vertexShaderCode = readSpirvFile("shaders/cube.vert.spv");
        auto fragmentShaderCode = readSpirvFile("shaders/cube.frag.spv");

        pipeline->setVertexShader(vertexShaderCode);
        pipeline->setFragmentShader(fragmentShaderCode);
        std::cout << "Shaders chargés" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erreur chargement shaders: " << e.what() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Configurer les vertex input
    pipeline->addVertexBinding(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);
    pipeline->addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos));
    pipeline->addVertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color));

    // Configurer le viewport
    pipeline->setViewport(800, 600);

    // Configurer le rasterizer SANS culling pour afficher toutes les faces
    pipeline->setRasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);

    // Activer le depth test
    pipeline->setDepthStencil(true, true, VK_COMPARE_OP_LESS);

    // Configurer le color blending
    pipeline->setColorBlendAttachment(false);

    // Push constants pour les matrices
    pipeline->addPushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ModelViewProj));

    // Ajouter les buffers à la tâche
    task.addVertexBuffer(&vertexBuffer);
    task.setIndexBuffer(&indexBuffer, VK_INDEX_TYPE_UINT32);
    task.setIndexedDrawParams(indices.size(), 1, 0, 0, 0);

    std::cout << "Construction du pipeline..." << std::endl;



    // Construire le pipeline
    if (!task.build(800, 600)) {
        std::cerr << "Erreur construction de la tâche GPU" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // ========================================================================
    // EXEMPLE : Utilisation AUTOMATIQUE de plusieurs pipelines
    // ========================================================================
    // Le système crée automatiquement des secondary command buffers si vous
    // avez plusieurs pipelines graphiques. Aucune configuration nécessaire !
    //
    // Exemple simple :
    /*
    // Pipeline 1 : Géométrie principale
    auto* geometryPipeline = task.createGraphicsPipeline("geometry");
    geometryPipeline->setVertexShader(...);
    geometryPipeline->setFragmentShader(...);
    // ... configuration ...
    
    // Pipeline 2 : Ombres
    auto* shadowPipeline = task.createGraphicsPipeline("shadows");
    shadowPipeline->setVertexShader(...);
    shadowPipeline->setFragmentShader(...);
    // ... configuration ...
    
    // Pipeline 3 : Particules
    auto* particlePipeline = task.createGraphicsPipeline("particles");
    particlePipeline->setVertexShader(...);
    particlePipeline->setFragmentShader(...);
    // ... configuration ...
    
    task.build();  // ← Crée automatiquement 3 secondary command buffers !
    task.execute(); // ← Tout s'exécute dans l'ordre automatiquement !
    */
    //
    // Le système détecte automatiquement qu'il y a plusieurs pipelines et :
    // 1. Crée un secondary command buffer par pipeline
    // 2. Les enregistre avec les bonnes commandes
    // 3. Les exécute dans l'ordre dans le primary command buffer
    //
    // Vous n'avez RIEN à faire, tout est automatique ! 🚀
    // ========================================================================

    // ========================================================================
    // EXEMPLE : Utilisation MANUELLE des Secondary Command Buffers (avancé)
    // ========================================================================
    // Les secondary command buffers permettent d'organiser le rendu en passes modulaires
    // Exemple : géométrie principale, ombres, post-processing, etc.

    // Décommentez cette section pour utiliser les secondary command buffers :
    /*
    std::cout << "\n=== Configuration des Secondary Command Buffers ===" << std::endl;

    // Créer un secondary command buffer pour la géométrie principale
    auto geometryBuffer = task.createSecondaryCommandBuffer("geometry");

    // Créer un secondary command buffer pour les ombres
    auto shadowBuffer = task.createSecondaryCommandBuffer("shadows");

    // Enregistrer les commandes pour la géométrie
    task.recordSecondaryCommandBuffer("geometry", [&](VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex) {
        auto* gfxPipeline = task.getGraphicsPipeline(0);
        if (!gfxPipeline) return;

        // Bind pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipeline->getPipeline());

        // Bind vertex buffers
        VkBuffer vb = vertexBuffer.getHandle();
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);

        // Bind index buffer
        vkCmdBindIndexBuffer(cmd, indexBuffer.getHandle(), 0, VK_INDEX_TYPE_UINT32);

        // Draw
        vkCmdDrawIndexed(cmd, indices.size(), 1, 0, 0, 0);
    });

    // Enregistrer les commandes pour les ombres (exemple)
    task.recordSecondaryCommandBuffer("shadows", [&](VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex) {
        // Ici vous pouvez ajouter le rendu des ombres
        std::cout << "Shadow pass for frame " << frameIndex << std::endl;
    });

    // Activer le mode custom recording pour utiliser les secondary buffers
    task.setUseCustomRecording(true);
    task.addRecordingCallback([&](VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex) {
        // Dans le primary command buffer, exécuter les secondary buffers
        task.beginDefaultRenderPass(cmd, imageIndex);
        task.executeSecondaryCommandBuffers(cmd);  // Exécute geometry + shadows
        task.endDefaultRenderPass(cmd);
    });

    std::cout << "Secondary command buffers configurés !" << std::endl;
    */
    // ========================================================================

    std::cout << "\n=== Render API initialisé avec succès! ===" << std::endl;
    std::cout << "Contrôles:" << std::endl;
    std::cout << "  ESC      - Quitter" << std::endl;
    std::cout << "  ESPACE   - Pause/Reprendre rotation" << std::endl;
    std::cout << "  FLECHES  - Ajuster distance caméra" << std::endl;
    std::cout << "\nLe cube devrait tourner automatiquement...\n" << std::endl;

    // Boucle principale
    bool running = true;
    bool paused = false;
    float cameraDistance = 3.0f;
    SDL_Event event;
    auto startTime = std::chrono::high_resolution_clock::now();
    float pausedTime = 0.0f;
    int frameCount = 0;
    auto lastFpsTime = startTime;

    while (running) {
        // Gérer les événements
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                } else if (event.key.keysym.sym == SDLK_SPACE) {
                    paused = !paused;
                    std::cout << (paused ? "Pause" : "Reprise") << std::endl;
                } else if (event.key.keysym.sym == SDLK_UP) {
                    cameraDistance -= 0.5f;
                    if (cameraDistance < 1.0f) cameraDistance = 1.0f;
                    std::cout << "Distance caméra: " << cameraDistance << std::endl;
                } else if (event.key.keysym.sym == SDLK_DOWN) {
                    cameraDistance += 0.5f;
                    if (cameraDistance > 10.0f) cameraDistance = 10.0f;
                    std::cout << "Distance caméra: " << cameraDistance << std::endl;
                }
            }
        }

        // Calculer le temps écoulé pour la rotation
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(currentTime - startTime).count();

        // Si en pause, garder l'angle constant
        if (paused) {
            pausedTime = time;
        }
        float angle = (paused ? pausedTime : time) * 0.8f; // Rotation à 0.8 rad/s

        // Créer les matrices
        ModelViewProj mvp;

        // Matrice modèle (rotation sur Y)
        matrixRotationY(mvp.model, angle);

        // Matrice vue (caméra reculée avec distance ajustable)
        matrixTranslation(mvp.view, 0.0f, 0.0f, -cameraDistance);

        // Matrice projection (perspective)
        matrixPerspective(mvp.proj, 3.14159f / 4.0f, 800.0f / 600.0f, 0.1f, 100.0f);

        // Mettre à jour les push constants
        task.pushConstants(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ModelViewProj), &mvp);

        // Exécuter le rendu
        // Note: Pas besoin de wait() car la synchronisation est gérée par les fences in-flight
        try {
            task.execute();
        } catch (const std::exception& e) {
            std::cerr << "Erreur rendu: " << e.what() << std::endl;
            break;
        }

        // Compter les FPS
        frameCount++;
        auto fpsDuration = std::chrono::duration<float>(currentTime - lastFpsTime).count();
        if (fpsDuration >= 1.0f) {
            float fps = frameCount / fpsDuration;
            std::cout << "FPS: " << static_cast<int>(fps) << std::endl;
            frameCount = 0;
            lastFpsTime = currentTime;
        }

        // Note: SDL_Delay retiré pour permettre un framerate illimité
        // Décommentez la ligne suivante pour limiter à ~60 FPS :
        // SDL_Delay(16);
    }

    std::cout << "\nNettoyage..." << std::endl;

    // Attendre que le GPU termine toutes les opérations avant le nettoyage
    vkDeviceWaitIdle(gpu->device);

    // Nettoyage
    task.destroy();
    vertexBuffer.destroy();
    indexBuffer.destroy();

    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cout << "Programme terminé proprement" << std::endl;

    // ========================================================================
    // INFO : Fonctionnalités avancées disponibles
    // ========================================================================
    //
    // 1. RECORDING PERSONNALISÉ :
    //    - task.setUseCustomRecording(true) - Active le mode personnalisé
    //    - task.addRecordingCallback() - Ajoute des callbacks de rendu
    //    - Callback signature : (VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex)
    //    - Voir: examples/custom_recording_example.cpp
    //
    // 2. SECONDARY COMMAND BUFFERS :
    //    - task.createSecondaryCommandBuffer(name) - Crée un secondary buffer
    //    - task.recordSecondaryCommandBuffer(name, callback) - Enregistre les commandes
    //    - task.executeSecondaryCommandBuffers(primaryCmd) - Exécute tous les secondary buffers
    //    - task.enableSecondaryCommandBuffer(name, bool) - Active/désactive un buffer
    //    - Permet d'organiser le rendu en passes modulaires (géométrie, ombres, etc.)
    //
    // 3. HELPERS :
    //    - task.beginDefaultRenderPass(cmd, imageIndex)
    //    - task.endDefaultRenderPass(cmd)
    //    - task.getGraphicsPipeline(index)
    //
    // Le mode automatique (actuel) est recommandé pour la plupart des cas.
    // Les secondary command buffers sont utiles pour des architectures de rendu complexes.
    // ========================================================================

    return 0;
}
