#pragma once
#include <string>

struct SynexisArguments {
    std::string modelPath;
    std::string modelProjectorPath;

    int numberOfGpuLayers = 999;
    int numberOfThreads = 4;
    bool use_mmap = true;

    int n_ctx = 16 * 1024;
    int n_batch = 1024;
    int n_keep = 512;
    int n_discard = 0;

    int n_slots = 8;

    bool embedding = false;

    explicit SynexisArguments(std::string modelPath): modelPath(std::move(modelPath)) {
    }
};
